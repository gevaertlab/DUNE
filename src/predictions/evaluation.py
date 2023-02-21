import logging
import torch
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, r2_score
from sksurv.metrics import brier_score
from matplotlib import pyplot as plt
from tqdm import tqdm
from nll_surv_loss import get_ci, NLLSurvLoss

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


def train_loop(model, dataloader, task, variable, num_classes, optimizer, device, train):

    loss_list = []
    output_list, labels_list = [] , []
    survival_months_list ,vital_status_list = [], []
    metrics_dict = {}

    if train:
        print("Train dataset...")
        model.train()
        for _, (features, labels, vital_status, survival_months) in enumerate(tqdm(dataloader, colour = "magenta")):
            inputs = features.to(device)
            labels = labels.to(device)
            outputs = model(inputs)


            optimizer.zero_grad()
            if task == "survival":
                vital_status = vital_status.to(device)
                
                if variable == "survival_bin":
                    censoring = 1 - vital_status
                    loss = NLLSurvLoss()(outputs, labels, censoring)

                else:
                    labels = labels.float()
                    surv_obj = (labels, vital_status)
                    loss = cox_loss(outputs, surv_obj)


            elif task == "regression" or num_classes == 2 :
                labels = labels.float()                
                outputs = outputs.view(-1)
                loss_func = MSELoss() if task == "regression" else BCELoss()
                loss = loss_func(outputs, labels)
                outputs = outputs > 0.5 if task == "classification" else outputs

            else: # task == "classification" and num_classes > 2
                ce_loss = CrossEntropyLoss()
                loss = ce_loss(outputs, labels)
                outputs = torch.argmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            vital_status_list.append(vital_status.detach().cpu().numpy())
            survival_months_list.append(survival_months.detach().cpu().numpy())
    

    else:
        print("\nValidation dataset...")
        model.eval()
        for _, (features, labels, vital_status, survival_months) in enumerate(tqdm(dataloader, colour = "cyan")):
            inputs = features.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            with torch.no_grad():
                if task == "survival":
                    vital_status = vital_status.to(device)

                    if variable == "survival_bin":
                        censoring = 1 - vital_status
                        loss = NLLSurvLoss()(outputs, labels, censoring)
                    else:
                        labels = labels.float()
                        surv_obj = (labels, vital_status)
                        loss = cox_loss(outputs, surv_obj)

                elif task == "regression" or num_classes == 2 :
                    labels = labels.float()                
                    outputs = outputs.view(-1)
                    loss_func = MSELoss() if task == "regression" else BCELoss()
                    loss = loss_func(outputs, labels)
                    outputs = outputs > 0.5 if task == "classification" else outputs

                else: # task == "classification" and num_classes > 2
                    ce_loss = CrossEntropyLoss()
                    loss = ce_loss(outputs, labels)
                    outputs = torch.argmax(outputs, dim=1)

            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
            vital_status_list.append(vital_status.detach().cpu().numpy())
            survival_months_list.append(survival_months.detach().cpu().numpy())

    output_list = np.concatenate(output_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    vital_status_list = np.concatenate(vital_status_list, axis=0)
    survival_months_list = np.concatenate(survival_months_list, axis=0)

    metrics_dict["loss"] = np.mean(loss_list)
    if task == "survival":
        if variable == "survival_bin":
            metrics_dict["concordance index"] = get_ci(
                output_list, vital_status_list, survival_months_list)
        else:
            metrics_dict["concordance index"] = concordance_index(
                labels_list, - output_list, vital_status_list)
            # metrics_dict['brier_score'] = brier_score(labels_list, output_list.flatten())


    elif task == "classification":
        metrics_dict["accuracy"] = accuracy_score(labels_list, output_list.flatten())
    else:
        metrics_dict["r2_score"] = r2_score(labels_list, output_list.flatten())

    return metrics_dict