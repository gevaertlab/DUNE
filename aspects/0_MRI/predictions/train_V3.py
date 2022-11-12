
import os
from os.path import join
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models import MROnlyModel
from datasets import MRDataset
from evaluation import evaluate, cox_loss
from utils import create_dependencies,parse_arguments, update_report, update_curves
plt.switch_backend('agg')




def train_model(model, dataloaders, optimizer, device, log_interval, config, output_dir):

    num_epochs = config["num_epochs"]
    task = config["task"]
    variable = config["variable"]
    best_val_loss = np.inf
    train_metrics = {}
    val_metrics = {}

    for epoch in range(num_epochs):
        model.train()

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        for b_idx, batch in enumerate(dataloaders['train']):

            inputs = batch['mr_data'].to(device)

            if task == "survival":
                survival_months = batch['survival_months'].to(device).float()
                vital_status = batch['vital_status'].to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))
                loss.backward()
                optimizer.step()
                vital_sum = vital_status.sum().item()


            elif task == "classification":
                labels = batch[variable].to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = CrossEntropyLoss()(outputs.view(-1), labels)
                loss.backward()
                optimizer.step()
                labels_sum = labels.sum().item()
                

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

    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x], num_workers=config["num_workers"]) for x in ['train', 'val', 'test']}

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

