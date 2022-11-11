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
    
    
def cox_loss(cox_scores, times, status):
    '''

    :param cox_scores: cox scores, size (batch_size)
    :param times: event times (either death or censor), size batch_size
    :param status: event status (1 for death, 0 for censor), size batch_size
    :return: loss of size 1, the sum of cox losses for the batch
    '''
            
    times, sorted_indices = torch.sort(-times)
    cox_scores = cox_scores[sorted_indices]
    status = status[sorted_indices]
    cox_scores = cox_scores -torch.max(cox_scores)
    exp_scores = torch.exp(cox_scores)
    loss = cox_scores - torch.log(torch.cumsum(exp_scores, dim=0)+1e-5)
    loss = - loss * status


    return loss.mean()
