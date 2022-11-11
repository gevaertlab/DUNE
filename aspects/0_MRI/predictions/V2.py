
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
import torch.multiprocessing
import torchvision
from lifelines.utils import concordance_index

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import os
from os.path import join
import json
import argparse
from models import cox_loss, MROnlyModel
from datasets import MRDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import create_dependencies,parse_arguments, update_report, update_curves
plt.switch_backend('agg')

TASK = "classification"
# Functions
###############


def evaluate(model, val_dataloader, task, variable, device, epoch):

    # Validation
    model.eval()

    output_list = []
    case_list = []
    loss_list = []
    
    

    if task == "survival":
        survival_months_list = []
        vital_status_list = []

        for _, batch_dict in enumerate(val_dataloader):
            inputs = batch_dict['mr_data'].to(device)
            # forward
            survival_months = batch_dict['survival_months'].to(device).float()
            vital_status = batch_dict['vital_status'].to(device).float()
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = cox_loss(
                    outputs.view(-1), survival_months.view(-1), vital_status.view(-1))

            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            survival_months_list.append(survival_months.detach().cpu().numpy())
            vital_status_list.append(vital_status.detach().cpu().numpy())
            case_list.append(batch_dict['case'])

        case_list = [c for c_b in case_list for c in c_b]
        survival_months_list = np.array(
            [s for s_b in survival_months_list for s in s_b])
        vital_status_list = np.array(
            [v for v_b in vital_status_list for v in v_b])

        output_list = np.concatenate(output_list, axis=0)

        case_CI, pandas_output = get_survival_CI(
            output_list, case_list, survival_months_list, vital_status_list)

        print(
            f"epoch {epoch} | CI {case_CI:.3f} | loss {np.mean(loss_list):.3f}")
        val_loss = np.mean(loss_list)

        metrics_dict = {"loss":val_loss, "concordance":case_CI}

    elif task == "classification":
        labels_list = []
        for _, batch_dict in enumerate(val_dataloader):
            inputs = batch_dict['mr_data'].to(device)
            labels = batch_dict[variable].to(device).float()
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1), labels)

            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            case_list.append(batch_dict['case'])
            labels_list.append(labels.detach().cpu().numpy())

        case_list = [c for c_b in case_list for c in c_b]
        labels_list = np.array([v for v_b in labels_list for v in v_b])
        output_list = np.concatenate(output_list, axis=0)

        val_loss = np.mean(loss_list)
        val_acc = accuracy_score(labels_list, output_list > .5)

        print(f"epoch {epoch} | acc {val_acc:.3f}")
        metrics_dict = {"loss":val_loss, "accuracy":val_acc}

    return metrics_dict


def get_survival_CI(output_list, ids_list, survival_months, vital_status):

    ids_unique = sorted(list(set(ids_list)))
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

    CI = concordance_index(survival_months_list, -
                           score_list, vital_status_list)
    pandas_output = pd.DataFrame({'id': ids_unique, 'score': score_list, 'survival_months': survival_months_list,
                                  'vital_status': vital_status_list})

    return CI, pandas_output


def train_model(model, dataloaders, optimizer, device, log_interval, config, output_dir):

    num_epochs = config["num_epochs"]
    task = config["task"]
    variable = config["variable"]

    best_val_loss = np.inf
    best_epoch = -1

    train_metrics = {}
    val_metrics = {}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        # TRAIN
        model.train()
        running_loss = 0.0
        inputs_seen = 0.0
        total_seen = 0.0

        # for logging
        last_running_loss = 0.0

        # Iterate over data.
        for b_idx, batch in enumerate(dataloaders['train']):

            inputs = batch['mr_data'].to(device)

            if task == "survival":
     
                survival_months = batch['survival_months'].to(device).float()
                vital_status = batch['vital_status'].to(device).float()
                # zero the parameter gradients
                optimizer.zero_grad()
                input_size = inputs.size()
                # forward
                outputs = model(inputs)
                loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))
                loss.backward()
                optimizer.step()
                vital_sum = vital_status.sum().item()
                # statistics
                running_loss += loss.item() * vital_sum
                inputs_seen += vital_sum
                total_seen += vital_sum


                if (b_idx+1 % log_interval == 0):
                    loss_to_log = (
                        running_loss - last_running_loss) / (inputs_seen)
                    last_running_loss = running_loss
                    inputs_seen = 0.0

                    print(
                        f"train | epoch {epoch} | batch {b_idx}/{len(dataloaders['train'])}| loss {loss_to_log:10.3f} "
                    )

            elif task == "classification":
                labels = batch[variable].to(device).float()
                optimizer.zero_grad()
                input_size = inputs.size()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1), labels)
                loss.backward()
                optimizer.step()
                labels_sum = labels.sum().item()
                running_loss += loss.item() * labels_sum
                inputs_seen += labels_sum
                total_seen += labels_sum


                if (b_idx+1 % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / (inputs_seen)
                    last_running_loss = running_loss
                    inputs_seen = 0.0
                    print(
                        f"train | epoch {epoch} | batch {b_idx}/{len(dataloaders['train'])}| loss {loss_to_log:10.3f}"
                    )

        epoch_loss = running_loss / total_seen
        print(f'TRAIN Loss: {epoch_loss:.4f}')

        train_epoch_metrics = evaluate(
            model, dataloaders['train'], task, variable, device, epoch)
        val_epoch_metrics = evaluate(
            model, dataloaders['val'], task, variable, device, epoch)

        metrics = [m for m in train_epoch_metrics.keys()]
        for m in metrics:
            if epoch == 0:
                train_metrics[m] = [train_epoch_metrics[m]]
                val_metrics[m] = [val_epoch_metrics[m]]
            else:
                train_metrics[m].append(train_epoch_metrics[m])
                val_metrics[m].append(val_epoch_metrics[m])


        if val_epoch_metrics['loss'] < best_val_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), join(
                output_dir, 'model_dict_best.pt'))
            best_val_loss = val_epoch_metrics['loss']


        report = update_report(
            output_dir, config, epoch, train_epoch_metrics, val_epoch_metrics)

        update_curves(report, metrics, output_dir)


    # Fin de la boucle d'entrainement
    torch.save(model.state_dict(), join(output_dir, 'model_last.pt'))
    print("\n")
    print("LOADING BEST MODEL, best epoch = {}".format(best_epoch))
    model.load_state_dict(torch.load(
        join(output_dir, 'model_dict_best.pt')))

    print("EVALUATING ON TEST SET")
    test_loss = evaluate(
        model, dataloaders['test'], task, variable, device, best_epoch)


def main():
    np.random.seed(333)
    torch.random.manual_seed(333)
    config = parse_arguments()
    variable = config["variable"]
    output_dir = create_dependencies(config["output_path"], variable)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MROnlyModel(10240, config['hidden_layer_size'])

    # Create training and validation datasets
    image_datasets = {}
    image_samplers = {}

    image_datasets['train'] = MRDataset(config["train_csv_path"])
    image_datasets['val'] = MRDataset(config["val_csv_path"])
    image_datasets['test'] = MRDataset(config["test_csv_path"])

    print("loaded datasets")
    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x], num_workers=config["num_workers"]) for x in ['train', 'val', 'test']}

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)
    if config['restore_path'] != "":
        model.load_state_dict(torch.load(config['restore_path']))
        print("Loaded model from checkpoint for finetuning")


    print("params to learn")
    params_to_update_rna = []
    params_to_update_mlp = []
    for n, param in model.mr_mlp.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_rna.append(param)
    for n, param in model.final_mlp.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_mlp.append(param)

    optimizer_ft = Adam([{'params': params_to_update_rna, 'lr': config['lr_rna']},
                        {'params': params_to_update_mlp, 'lr': config['lr_mlp']}],
                        weight_decay=config['weight_decay'])

    # Train and evaluate

    train_model(model=model, dataloaders=dataloaders_dict,
                optimizer=optimizer_ft,
                device=device,
                log_interval=100,
                config=config,
                output_dir=output_dir)


# MAIN
##########

if __name__ == '__main__':
    main()


# REGRESSION ET LEARNING RATES !!!!!
# BACKUP MODEL