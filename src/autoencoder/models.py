import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset


class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4, num_blocks=5, unet=False):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.UNet = unet
        self.s = 2 if unet else 1
        
        # ENCODER BLOCKS
        feature_list = [init_features*(2**n) for n in range(num_blocks)]
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        for i, n in enumerate(feature_list):
            if i ==0:
                enc_block = VAE._block(self.in_channels, n, name=f"enc{i}")
            else:
                enc_block = VAE._block(feature_list[i-1], n, name=f"enc{i}")

            self.enc_blocks.append(enc_block)
        
        # BOTTLENECK EXTRACTION
        bottleneck_features= 2*feature_list[-1]
        self.bottleneck = VAE._block(feature_list[-1], bottleneck_features, name="bottleneck")

        # DECODER BLOCKS
        feature_list = feature_list[::-1]
        self.dec_blocks = nn.ModuleList()
        self.transposers = nn.ModuleList()
        for i, n in enumerate(feature_list):
            if i==0:
                upconv = nn.ConvTranspose3d(bottleneck_features, n, kernel_size=2, stride=2)
                dec_block = VAE._block(n*self.s, n, name=f"dec{i}")
            else: 
                upconv = nn.ConvTranspose3d(feature_list[i-1], n, kernel_size=2, stride=2)
                dec_block = VAE._block(n*self.s, n, name=f"dec{i}")
            
            self.transposers.append(upconv)
            self.dec_blocks.append(dec_block)
            
        # FINAL CONVOLUTION
        self.last_conv = nn.Conv3d(feature_list[-1], out_channels=out_channels, kernel_size=1)


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


            if self.UNet:
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

