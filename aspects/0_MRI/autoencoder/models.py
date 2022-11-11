import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder5 = UNet3D._block(features * 8, features * 16, name="enc4")
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(
            features * 16, features * 32, name="bottleneck")

        self.upconv5 = nn.ConvTranspose3d(
            features * 32, features * 16, kernel_size=2, stride=2
        )

        self.decoder5 = UNet3D._block(
            (features * 16) * 2, features * 16, name="dec4")
        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )

        self.decoder4 = UNet3D._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet3D._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet3D._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet3D._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.upconv5(bottleneck, output_size=enc5.shape)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5, output_size=enc4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4, output_size=enc3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3, output_size=enc2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2, output_size=enc1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1)), bottleneck

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class convAE(nn.Module):
    def __init__(self, in_c, out_c, num_feat, num_blocks):
        super().__init__()

        # Creating the neural network structure
        self.lastOut = []
        self.num_blocks = num_blocks

        # Encoder tools
        self.conv1 = nn.Conv3d(
            in_channels=in_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=out_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, stride=2, padding=0, return_indices=True)

        # Feature extraction layer
        self.extr = nn.Conv3d(
            in_channels=out_c, out_channels=num_feat, kernel_size=[3, 3, 3], stride=1, padding=1)

        # Decoder tools
        self.unpool = nn.MaxUnpool3d(2, stride=2, padding=0)
        self.start_decode = nn.ConvTranspose3d(
            in_channels=num_feat, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv2 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=in_c, kernel_size=[3, 3, 3], stride=1, padding=1)

    def encoder(self, features):
        indexes, shapes = [], [features.size()]
        for k in range(self.num_blocks):
            if k == 0:
                x = func.relu(self.conv1(features))
            else:
                x = func.relu(self.conv2(x))

            x = func.relu(self.conv2(x))
            x, idk = self.pool(x)
            indexes.append(idk)
            shapes.append(x.size())

        coded_img = self.extr(x)

        return coded_img, indexes, shapes

    def decoder(self, x, indexes, shapes):
        indexes.reverse()
        shapes.reverse()

        x = self.start_decode(x)
        for k in range(self.num_blocks):
            x = self.unpool(x, indexes[k], output_size=shapes[k+1])
            x = func.relu(self.t_conv2(x))

            if k != self.num_blocks-1:
                x = func.relu(self.t_conv2(x))
            else:
                decoded_img = func.relu(self.t_conv1(x))

        decoded_img = decoded_img / torch.max(decoded_img)

        return decoded_img

    def forward(self, features):

        coded_img, indexes, shapes = self.encoder(features)
        decoded_img = self.decoder(coded_img, indexes, shapes)

        return decoded_img, code

