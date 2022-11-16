import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, num_blocks=5):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for i, n in enumerate(feature_list):
            if i ==0:
                enc_block = UNet3D._block(self.in_channels, n, name=f"enc{i}")
            else:
                enc_block = UNet3D._block(feature_list[i-1], n, name=f"enc{i}")

            self.enc_blocks.append(enc_block)
        
        # BOTTLENECK EXTRACTION
        bottleneck_features= 2*n
        self.bottleneck = UNet3D._block(n, bottleneck_features, name="bottleneck")

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.dec_blocks = nn.ModuleList()
        self.transposers = nn.ModuleList()
        for i, n in enumerate(feature_list):
            if i==0:
                upconv = nn.ConvTranspose3d(bottleneck_features, n, kernel_size=2, stride=2)
                dec_block = UNet3D._block(bottleneck_features, n, name=f"dec{i}")
            else: 
                upconv = nn.ConvTranspose3d(feature_list[i-1], n, kernel_size=2, stride=2)
                dec_block = UNet3D._block(feature_list[i-1], n, name=f"dec{i}")
            
            self.transposers.append(upconv)
            self.dec_blocks.append(dec_block)

        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(n, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        ## ENCODING
        encodings = []
        for k in range(self.num_blocks):
            enc = self.enc_blocks[k](x)
            x = self.pool(enc)
            encodings.append(enc)
            
        ## REPRESENTATION EXTRACTION
        bottleneck = self.bottleneck(x)

        ## DECODING
        encodings.reverse()
        for k in range(self.num_blocks):   
            if k==0:
                dec = self.transposers[k](bottleneck, output_size= encodings[k].shape)
            else:
                dec = self.transposers[k](dec, output_size= encodings[k].shape)

            dec = torch.cat((dec, encodings[k]), dim=1)
            dec = self.dec_blocks[k](dec)

        return torch.sigmoid(self.last_conv(dec)), bottleneck

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







