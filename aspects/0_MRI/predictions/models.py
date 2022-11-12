from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MROnlyModel(nn.Module):
    def __init__(self, num_features=10240, hidden_layer_size=2048):
        super(MROnlyModel, self).__init__()

        self.mr_mlp = torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_features, hidden_layer_size),
            nn.ReLU(),
        )

        self.final_mlp = torch.nn.Sequential(nn.Linear(2048, 1))



    def forward(self, features):
        x = self.mr_mlp(features)
        x = self.final_mlp(x)
        return x
    
    def extract(self,features):
        x = self.mr_mlp(features)
        return x
    
    