import torch

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
        bottleneck_features= 2*feature_list[-1]
        self.bottleneck = UNet3D._block(feature_list[-1], bottleneck_features, name="bottleneck")

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

        return bottleneck

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




class CompleteRegressor(nn.Module):
    def __init__(
        self, encoder, loss_fn, n_inputs: int = 1, l1_lambda=0.05, l2_lambda=0.05, classifier=True
    ):
        super(CompleteRegressor, self).__init__()
        self.encoder = encoder
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.train_log = []

    def forward(self, x):
        x = self.encoder(x).view(5, -1)
        outputs = self.output_layer(x)

        if self.classifier:
            outputs=torch.sigmoid(outputs)
        return outputs

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()

        return self.l2_lambda * l2_norm

    def compute_loss(self, x, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()

        return loss
    


class ElasticLinear(nn.Module):
    def __init__(
        self, loss_fn, n_inputs: int = 1, num_class=1, l1_lambda=0.05, l2_lambda=0.05, classifier=True
    ):
        super(ElasticLinear, self).__init__()
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.classifier= classifier
        self.hidden = torch.nn.Linear(n_inputs, 256)
        self.hidden2 = torch.nn.Linear(256, 16)
        self.output_layer = torch.nn.Linear(16, num_class)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden2(x)
        outputs = self.output_layer(x)

        if self.classifier:
            outputs=torch.nn.functional.softmax(outputs)

        return outputs

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()

        return self.l2_lambda * l2_norm

    def compute_loss(self, x, y):
        dt = torch.long if self.classifier else torch.float32
        y = torch.tensor(y, dtype=dt)
        # x = torch.tensor(x, dtype=torch.float32)
        y_hat = self(x)
        y_hat = torch.squeeze(y_hat)
        loss = self.loss_fn(y_hat, y)  + self.l1_reg() + self.l2_reg()


        return loss



# class ElasticLinear(pl.LightningModule):
#     def __init__(
#         self, encoder, loss_fn, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.learning_rate = learning_rate
#         self.loss_fn = loss_fn
#         self.l1_lambda = l1_lambda
#         self.l2_lambda = l2_lambda
#         self.output_layer = torch.nn.Linear(n_inputs, 1)
#         self.train_log = []

#     def forward(self, x):
#         x = self.encoder(x)
#         outputs = self.output_layer(x)
#         return outputs

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         return optimizer

#     def l1_reg(self):
#         l1_norm = self.output_layer.weight.abs().sum()

#         return self.l1_lambda * l1_norm

#     def l2_reg(self):
#         l2_norm = self.output_layer.weight.pow(2).sum()

#         return self.l2_lambda * l2_norm

#     def training_(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()

#         self.log("loss", loss)
#         self.train_log.append(loss.detach().numpy())
#         return loss
    
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()

#         self.log("test loss", loss)
#         return loss



