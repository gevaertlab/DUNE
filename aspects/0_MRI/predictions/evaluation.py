import torch
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score


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
    cox_scores = cox_scores - torch.max(cox_scores)
    exp_scores = torch.exp(cox_scores)
    loss = cox_scores - torch.log(torch.cumsum(exp_scores, dim=0)+1e-5)
    loss = - loss * status

    return loss.mean()


def get_survival_CI(output_list, ids_list, survival_months, vital_status):

    # ids_unique = sorted(list(set(ids_list)))
    id_to_scores = {}
    id_to_survival_months = {}
    id_to_vital_status = {}

    for i in range(len(output_list)):
        id = ids_list[i]
        id_to_scores[id] = id_to_scores.get(id, []) + [output_list[i, 0]]
        id_to_survival_months[id] = survival_months[i]
        id_to_vital_status[id] = vital_status[i]

    for k in id_to_scores.keys():
        id_to_scores[k] = np.mean(id_to_scores[k])

    score_list = np.array([id_to_scores[id] for id in ids_unique])
    survival_months_list = np.array(
        [id_to_survival_months[id] for id in ids_unique])
    vital_status_list = np.array([id_to_vital_status[id] for id in ids_unique])

    CI = concordance_index(survival_months_list, - score_list, vital_status_list)

    return CI


def evaluate(model, val_dataloader, task, num_classes, device, epoch):

    # Validation
    model.eval()

    output_list = []
    labels_list = []
    loss_list = []
    metrics_dict = {}

    for _, (features, labels) in enumerate(val_dataloader):
        inputs = features.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)


        with torch.no_grad():
            if task == "survival":
                # survival_months_list = []
                # vital_status_list = []

                # inputs = batch_dict['mr_data'].to(device)
                # # forward
                # survival_months = batch_dict['survival_months'].to(device).float()
                # vital_status = batch_dict['vital_status'].to(device).float()
                # with torch.no_grad():
                #     outputs = model.forward(inputs)
                #     loss = cox_loss(
                #         outputs.view(-1), survival_months.view(-1), vital_status.view(-1))

                # loss_list.append(loss.item())
                # output_list.append(outputs.detach().cpu().numpy())
                # survival_months_list.append(survival_months.detach().cpu().numpy())
                # vital_status_list.append(vital_status.detach().cpu().numpy())
                # case_list.append(batch_dict['case'])

                # case_list = [c for c_b in case_list for c in c_b]
                # survival_months_list = np.array(
                #     [s for s_b in survival_months_list for s in s_b])
                # vital_status_list = np.array(
                #     [v for v_b in vital_status_list for v in v_b])

                # output_list = np.concatenate(output_list, axis=0)

                # case_CI, pandas_output = get_survival_CI(
                #     output_list, case_list, survival_months_list, vital_status_list)

                # print(
                #     f"epoch {epoch} | CI {case_CI:.3f} | loss {np.mean(loss_list):.3f}")

                # val_loss = np.mean(loss_list)
                # metrics_dict = {"loss": val_loss, "concordance": case_CI}
                pass


            elif task == "classification":
                if num_classes > 2:
                    loss_func = CrossEntropyLoss()
                    loss = loss_func(outputs, labels)
                    outputs = torch.argmax(outputs, dim=1)
                else:
                    loss_func = BCELoss()
                    loss = loss_func(outputs.view(-1), labels.float())
                    outputs = outputs > 0.5
                                    
            elif task == "regression":
                    loss_func = MSELoss()
                    loss = loss_func(outputs.view(-1), labels.float())


        loss_list.append(loss.item())   
        output_list.append(outputs.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    labels_list = np.concatenate(labels_list, axis=0)
    output_list = np.concatenate(output_list, axis=0)

    metrics_dict["loss"] = np.mean(loss_list)
    metrics_dict["loss2"] = np.mean(loss_list)
    # metrics_dict["accuracy"] = accuracy_score(labels_list, output_list)
    

    return metrics_dict
