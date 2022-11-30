import torch
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, r2_score
from matplotlib import pyplot as plt


def cox_loss(cox_scores, labels):
    '''
    :param cox_scores: cox scores, size (batch_size)
    :param times: event times (either death or censor), size batch_size
    :param status: event status (1 for death, 0 for censor), size batch_size
    :return: loss of size 1, the sum of cox losses for the batch
    '''
    times, status = labels
    times, sorted_indices = torch.sort(-times)
    cox_scores = cox_scores[sorted_indices]
    status = status[sorted_indices]
    cox_scores = cox_scores - torch.max(cox_scores)
    exp_scores = torch.exp(cox_scores)
    loss = cox_scores - torch.log(torch.cumsum(exp_scores, dim=0)+1e-5)
    loss = - loss * status

    return loss.mean()


def evaluate(model, val_dataloader, task, num_classes, device):

    model.eval()

    loss_list = []
    output_list, labels_list, vital_status_list = [], [], []
    metrics_dict = {}

    for _, (features, labels, vital_status) in enumerate(val_dataloader):
        inputs = features.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)

        with torch.no_grad():
            if task == "survival":
                labels = labels.float()
                vital_status = vital_status.to(device)
                surv_obj = (labels, vital_status)
                loss = cox_loss(outputs, surv_obj)

            elif task == "classification":
                if num_classes > 2:
                    loss_func = CrossEntropyLoss()
                    loss = loss_func(outputs, labels)
                    outputs = torch.argmax(outputs, dim=1)
                else:
                    loss_func = BCELoss()
                    loss = loss_func(outputs.view(-1), labels.float())
                    outputs = outputs > 0.5

            else:  # task == "regression":
                loss_func = MSELoss()
                loss = loss_func(outputs.view(-1), labels.float())

        loss_list.append(loss.item())
        output_list.append(outputs.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        vital_status_list.append(vital_status.detach().cpu().numpy())

    output_list = np.concatenate(output_list, axis=0).flatten()
    labels_list = np.concatenate(labels_list, axis=0)
    vital_status_list = np.concatenate(vital_status_list, axis=0)

    metrics_dict["loss"] = np.mean(loss_list)
    if task == "survival":
        metrics_dict["concordance index"] = concordance_index(
            labels_list, - output_list, vital_status_list)
    elif task == "classification":
        metrics_dict["accuracy"] = accuracy_score(labels_list, output_list)
    else:
        metrics_dict["r2_score"] = r2_score(labels_list, output_list)

    return metrics_dict