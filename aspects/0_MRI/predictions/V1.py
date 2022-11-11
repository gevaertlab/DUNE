
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import RandomSampler
import torch.multiprocessing
import torchvision
# from torchvision.utils import *
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
plt.switch_backend('agg')

TASK = "classification"
# Functions
###############


def evaluate(model, val_dataloader, task, device, epoch, mode='val'):

    # Validation
    model.eval()

    output_list = []
    case_list = []
    loss_list = []
    survival_months_list = []
    vital_status_list = []
    sex_list = []

    if task == "survival_prediction":

        for _, batch_dict in enumerate(val_dataloader):
            inputs = batch_dict['mr_data'].to(device)

            # forward

            survival_months = batch_dict['survival_months'].to(device).float()
            vital_status = batch_dict['vital_status'].to(device).float()
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = cox_loss(outputs.view(-1), survival_months.view(-1), vital_status.view(-1))

            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            survival_months_list.append(survival_months.detach().cpu().numpy())
            vital_status_list.append(vital_status.detach().cpu().numpy())
            case_list.append(batch_dict['case'])

        
        case_list = [c for c_b in case_list for c in c_b]
        survival_months_list = np.array(
                    [s for s_b in survival_months_list for s in s_b])
        vital_status_list = np.array([v for v_b in vital_status_list for v in v_b])

        output_list = np.concatenate(output_list, axis=0)

        case_CI, pandas_output = get_survival_CI(
                    output_list, case_list, survival_months_list, vital_status_list)

        print(f"{mode} case  | epoch {epoch} | CI {case_CI:.3f} | loss {np.mean(loss_list):.3f}")
        val_loss = np.mean(loss_list)

        return val_loss, case_CI

        
    elif task=="classification":
        for _, batch_dict in enumerate(val_dataloader):
            inputs = batch_dict['mr_data'].to(device)
            sex = batch_dict['sex'].to(device).float()
            with torch.no_grad():
                outputs = model.forward(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1), sex)
            
            loss_list.append(loss.item())
            output_list.append(outputs.detach().cpu().numpy())
            case_list.append(batch_dict['case'])

            sex_list.append(sex.detach().cpu().numpy())

            
        
        case_list = [c for c_b in case_list for c in c_b]
        sex_list = np.array([v for v_b in sex_list for v in v_b])
        output_list = np.concatenate(output_list, axis=0)
        val_acc = accuracy_score(sex_list, output_list > .5)
        print(f"{mode} case  | epoch {epoch} | acc {val_acc:.3f}")

        val_loss = np.mean(loss_list)




        return val_loss, val_acc


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


def train_model(model, dataloaders, task, optimizer, device, num_epochs=25,
                log_interval=100, save_dir='checkpoints/models'):

    best_val_loss = np.inf
    best_epoch = -1

    Val_losses, Train_losses = [], []
    Val_accs, Train_accs = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            if task=="survival_prediction":

                survival_months = batch['survival_months'].to(device).float()
                vital_status = batch['vital_status'].to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()
                input_size = inputs.size()

                # forward
                outputs = model(inputs)
                loss = cox_loss(outputs.view(-1),
                                survival_months.view(-1), vital_status.view(-1))
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

                    last_time = time.time()

                    last_running_loss = running_loss
                    inputs_seen = 0.0

                    print(
                        f"train | epoch {epoch} | batch {b_idx}/{len(dataloaders['train'])}| loss {loss_to_log:10.3f} "
                    )
        
            elif task == "classification":
                sex = batch['sex'].to(device).float()
                optimizer.zero_grad()
                input_size = inputs.size()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs.view(-1), sex)
                loss.backward()
                optimizer.step()
                sex_sum = sex.sum().item()
                running_loss += loss.item() * sex_sum
                inputs_seen += sex_sum
                total_seen += sex_sum
                if (b_idx+1 % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / (inputs_seen)
                    last_time = time.time()
                    last_running_loss = running_loss
                    inputs_seen = 0.0
                    print(
                        f"train | epoch {epoch} | batch {b_idx}/{len(dataloaders['train'])}| loss {loss_to_log:10.3f} "
                    )


        epoch_loss = running_loss / total_seen
        print(f'TRAIN Loss: {epoch_loss:.4f}')

        train_loss, train_acc = evaluate(
            model, dataloaders['train'],TASK,  device, epoch, mode='train')
        val_loss, val_acc  = evaluate(
            model, dataloaders['val'], TASK, device, epoch, mode='val')

        if val_loss < best_val_loss:
            best_epoch = epoch
            torch.save(model.state_dict(), join(
            save_dir, 'model_dict_best.pt'))
            best_val_loss = val_loss

        Train_losses.append(train_loss)       
        Val_losses.append(val_loss)
        Train_accs.append(train_acc)
        Val_accs.append(val_acc)

        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(7, 10))
        ax[0].plot(Train_losses, label="Train")
        ax[0].plot(Val_losses, label="Val")
        ax[0].set_title("Loss")
        ax[1].plot(Train_accs, label="Train")
        ax[1].plot(Val_accs, label="Val")
        ax[1].set_title("Accuracy")
        ax[0].legend()
        fig.savefig("curves.png")
        plt.close(fig)

        # plt.figure()
        # plt.plot(Train_losses, label='Train')
        # plt.plot(Val_losses, label='Val')
        # plt.legend()
        # plt.close()


    torch.save(model.state_dict(), join(save_dir, 'model_last.pt'))


    print("\n")
    print("LOADING BEST MODEL, best epoch = {}".format(best_epoch))
    model.load_state_dict(torch.load(
        join(save_dir, 'model_dict_best.pt')))

    print("EVALUATING ON TEST SET")
    test_loss  = evaluate(
        model, dataloaders['test'], TASK,device, best_epoch, mode='test')


def main():

    args = parser.parse_args()
    np.random.seed(333)
    torch.random.manual_seed(333)
    torch.multiprocessing.set_sharing_strategy('file_system')

    with open(args.config) as f:
        config = json.load(f)
    if 'flag' in config:
        args.flag = config['flag']
    if 'checkpoint_path' in config:
        args.checkpoint_path = config['checkpoint_path']
    if 'summary_path' in config:
        args.summary_path = config['summary_path']
    if args.flag == "":
        args.flag = 'train_coxloss_{date:%Y-%m-%d %H:%M:%S}'.format(
            date=datetime.datetime.now())

    device = torch.device("cuda" if (
        torch.cuda.is_available() and config['use_cuda']) else "cpu")
    num_epochs = config['num_epochs']

    model_rna = torch.nn.Sequential(
        nn.Dropout(),
        nn.Linear(10240, 2048),  # INPUT SIZE A CHANGER ICI
        nn.ReLU(),

    )

    combine_mlp = torch.nn.Sequential(nn.Linear(2048, 1))
    model = MROnlyModel(model_rna, combine_mlp)

    print("Loaded model")

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
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=config['batch_size'], sampler=image_samplers[x], num_workers=config["num_workers"])
        for x in ['train', 'val', 'test']
        }

    print("Initialized Datasets and Dataloaders...")

    # Send the model to GPU
    model = model.to(device)
    if config['restore_path'] != "":
        model.load_state_dict(torch.load(config['restore_path']))
        print("Loaded model from checkpoint for finetuning")

    params_to_update_rna = []
    params_to_update_mlp = []

    print("params to learn")

    for n, param in model_rna.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_rna.append(param)
    for n, param in combine_mlp.named_parameters():
        if param.requires_grad:
            print("\t {}".format(n))
            params_to_update_mlp.append(param)

    optimizer_ft = Adam([{'params': params_to_update_rna, 'lr': config['lr_rna']},
                        {'params': params_to_update_mlp, 'lr': config['lr_mlp']}],
                        weight_decay=config['weight_decay'])

    # Train and evaluate

    os.makedirs(join(args.checkpoint_path, 'models', args.flag), exist_ok=True)

    train_model(model=model, dataloaders=dataloaders_dict,
                task = TASK,
                optimizer=optimizer_ft,
                device=device,
                num_epochs=num_epochs,
                save_dir=join(args.checkpoint_path, 'models', args.flag))

# Input arguments
####################


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='config.json', help='configuration json file')

# MAIN
##########

if __name__ == '__main__':
    main()
