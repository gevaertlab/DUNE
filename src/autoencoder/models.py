import os
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset


class VAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4, num_blocks=5, type_ae="VAE"):
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.UNet = True if type_ae=="UNet" else False
        self.s = 2 if self.UNet else 1
        
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


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        self.mode = mode
        if mode == 'encode':
            self.conv1 = nn.Conv3d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv3d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose3d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose3d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm3d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x, shape=None):

        if self.mode=="encode":
            conv1 = self.BN(self.conv1(x))
            relu = self.relu(conv1)
            conv2 = self.BN(self.conv2(relu))
            if self.resize:
                x = self.BN(self.conv1(x))

        elif self.mode=="decode":
            conv1 = self.BN(self.conv1(x, output_size=shape))
            relu = self.relu(conv1)
            conv2 = self.BN(self.conv2(relu))

            if self.resize:
                x = self.BN(self.conv1(x, output_size=shape))

        return self.relu(x + conv2), list(self.relu(x+conv2).size())
    

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self, NMOD):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv3d(NMOD, 4, 2, 1, 1) # 16 32 32
        self.BN = nn.BatchNorm3d(4)
        self.rb1 = ResBlock(4, 4, 2, 2, 1, 'encode') # 16 16 16
        self.rb2 = ResBlock(4, 8, 2, 2, 1, 'encode') # 32 16 16
        self.rb3 = ResBlock(8, 8, 2, 2, 1, 'encode') # 32 8 8
        self.rb4 = ResBlock(8, 16, 2, 2, 1, 'encode') # 48 8 8
        self.rb5 = ResBlock(16, 16, 2, 2, 1, 'encode') # 48 4 4
        self.rb6 = ResBlock(16, 32, 2, 2, 1, 'encode') # 64 2 2
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1, rb1_shape = self.rb1(init_conv)
        rb2, rb2_shape = self.rb2(rb1)
        rb3, rb3_shape = self.rb3(rb2)
        rb4, rb4_shape = self.rb4(rb3)
        rb5, rb5_shape = self.rb5(rb4)
        rb6, rb6_shape = self.rb6(rb5)

        return rb6, [rb6_shape, rb5_shape, rb4_shape, rb3_shape, rb2_shape, rb1_shape, list(init_conv.size()), list(inputs.size())]

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self, NMOD):
        super(Decoder, self).__init__()
        self.rb6 = ResBlock(32, 16, 2, 2, 1, 'decode') # 48 4 4
        self.rb5 = ResBlock(16, 16, 2, 2, 1, 'decode') # 48 8 8
        self.rb4 = ResBlock(16, 8, 2, 2, 1, 'decode') # 32 8 8
        self.rb3 = ResBlock(8, 8, 2, 2, 1, 'decode') # 32 16 16
        self.rb2 = ResBlock(8, 4, 2, 2, 1, 'decode') # 16 16 16
        self.rb1 = ResBlock(4, 4, 2, 2, 1, 'decode') # 16 32 32
        self.out_conv = nn.ConvTranspose3d(4, NMOD, 2, 1, 1) # 3 32 32
        self.tanh = nn.Tanh()
        
    def forward(self, inputs, shapes):
        rb6, _ = self.rb6(inputs, shapes[1])
        rb5, _ = self.rb5(rb6, shapes[2])
        rb4, _ = self.rb4(rb5, shapes[3])
        rb3, _ = self.rb3(rb4, shapes[4])
        rb2, _ = self.rb2(rb3, shapes[5])
        rb1, _ = self.rb1(rb2, shapes[6])
        out_conv = self.out_conv(rb1, output_size=shapes[7])
        output = self.tanh(out_conv)
        return output
    

class RNet(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """

    def __init__(self, NMOD=3):
        super(RNet, self).__init__()
        self.encoder = Encoder(NMOD)
        self.decoder = Decoder(NMOD)

    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p

    def forward(self, inputs):
        encoded, shapes = self.encoder(inputs)
        decoded = self.decoder(encoded, shapes)
        decoded = torch.sigmoid(decoded)
        return decoded, encoded
    
# https://github.com/jan-xu/autoencoders/blob/master/resnet/resnet.ipynb