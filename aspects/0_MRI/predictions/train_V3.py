
import os
from os.path import join
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from models import MROnlyModel
from datasets import MRDataset
from evaluation import evaluate, cox_loss
from utils import create_dependencies,parse_arguments, update_report, update_curves
plt.switch_backend('agg')


def train_model(model, dataloaders, optimizer, device, log_interval, config, output_dir):

    num_epochs = config["num_epochs"]
    best_epoch = 0
    task = config["task"]
    num_classes = config['num_classes']
    best_val_loss = np.inf
    train_metrics = {}
    val_metrics = {}

    for epoch in range(num_epochs):
        model.train()

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        for _, (features, labels) in enumerate(dataloaders['train']):

            inputs = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == "survival":
                # survival_months = batch['survival_months'].to(device).float()
                # vital_status = batch['vital_status'].to(device).float()
                # loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))
                # loss.backward()
                # optimizer.step()
                # vital_sum = vital_status.sum().item()
                pass


            elif task == "regression" or num_classes == 2 :
                labels = labels.float()                
                outputs = outputs.view(-1)
                loss_func = MSELoss() if task == "regression" else BCELoss()

            elif task == "classification" and num_classes > 2:
                loss_func = CrossEntropyLoss()
                    
        
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()                

        train_epoch_metrics = evaluate(
            model, dataloaders['train'], task, num_classes, device, epoch)
        val_epoch_metrics = evaluate(
            model, dataloaders['val'], task, num_classes, device, epoch)

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
        model, dataloaders['test'], task, device, best_epoch)


def main():
    np.random.seed(333)
    torch.random.manual_seed(333)
    config = parse_arguments()
    variable = config["variable"]
    output_dir = create_dependencies(config["output_path"], variable)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Create training and validation datasets
    image_datasets = {}
    image_samplers = {}
    train_csv_path = config["csv_paths"]+"/concat_train.csv"
    val_csv_path = config["csv_paths"]+"/concat_val.csv"
    test_csv_path = config["csv_paths"]+"/concat_test.csv"

    image_datasets['train'] = MRDataset(train_csv_path, variable, config["task"])
    image_datasets['val'] = MRDataset(val_csv_path,variable, config["task"])
    image_datasets['test'] = MRDataset(test_csv_path, variable, config["task"])

    image_samplers['train'] = RandomSampler(image_datasets['train'])
    image_samplers['val'] = RandomSampler(image_datasets['val'])
    image_samplers['test'] = RandomSampler(image_datasets['test'])

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: DataLoader(image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x], num_workers=config["num_workers"]) for x in ['train', 'val', 'test']}

    # Send the model to GPU
    config['num_classes'] = image_datasets['train'].num_classes
    model = MROnlyModel(config['num_classes'], config["task"], config['num_features'], config['hidden_layer_size'])
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

