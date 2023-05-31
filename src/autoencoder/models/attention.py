from abc import abstractmethod
from torch import nn
from torch import tensor as Tensor
import torch
from typing import List, Any


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, depth * height * width)
        key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        value = self.value_conv(x).view(batch_size, -1, depth * height * width)

        attention_scores = torch.matmul(query, key.permute(0, 2, 1))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.view(batch_size, channels, depth, height, width)

        out = self.gamma * attention_output + x
        return out
