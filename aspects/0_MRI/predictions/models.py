import torch
import torch.nn as nn


class MROnlyModel(nn.Module):
    def __init__(self, task, num_classes, num_features, hidden_layer_size):
        super(MROnlyModel, self).__init__()
        self.num_classes = 1 if num_classes == 2 else num_classes
        self.task = task
        self.hidden_layer_size = hidden_layer_size
        self.num_features = num_features

        self.mr_mlp = torch.nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.num_features, self.hidden_layer_size),
            nn.ReLU(),
        )

        self.final_mlp = torch.nn.Sequential(nn.Linear(self.hidden_layer_size, self.num_classes))

        


    def forward(self, features):
        x = self.mr_mlp(features)
        x = self.final_mlp(x)

        if self.task == "classification" and self.num_classes == 1:
            x = nn.Sigmoid()(x)

        return x
    
    def extract(self,features):
        x = self.mr_mlp(features)
        return x
    


