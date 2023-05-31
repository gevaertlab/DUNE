import yaml
import argparse
import os
from os.path import join, exists
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import torch
from torchvision.utils import save_image, make_grid
import numpy as np

from torchvision import transforms as transforms
from datasets import BrainImages
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split


from models.oldmodels import OldAE
from models.classicalAE import BaseAE
from models.resnetAE import RNet
from models.VAEs import U_VAE, VAE3D

# %% Init


def create_dependencies(model_path):

    os.makedirs(join(model_path, "exports/features"), exist_ok=True)
    os.makedirs(join(model_path, "logs"), exist_ok=True)
    os.makedirs(join(model_path, "multivariate/models"), exist_ok=True)
    os.makedirs(join(model_path, "multivariate/conf_mat"), exist_ok=True)


def parse_arguments(exp):
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file',
                        type=str,
                        help='config file path')
    parser.add_argument('-o', '--output_name', type=str,
                        help='output_name - for predictions only', required=False)
    parser.add_argument('-f','--features', type=str,
                        help='output_name - for predictions only', required=False)

    args = parser.parse_args()

    config_file = join("outputs", args.config_file, "config.cfg")

    print(f"Importing {config_file}...")
    conf_parser = configparser.ConfigParser()
    conf_parser.read(config_file)

    conf = {k:eval(v) for k, v in dict(conf_parser["config"]).items()}
    model = {k:eval(v) for k, v in dict(conf_parser["model"]).items()}
    data = {k:eval(v) for k, v in dict(conf_parser["data"]).items()}
    predictions = {k:eval(v) for k, v in dict(conf_parser["predictions"]).items()}
    model["model_path"] = join(model["model_path"], model['model_name'])

    

    create_dependencies(model["model_path"])


    if exp == "ae":
        export = [{"model":model, "data":data, "config":conf }]
        config = {**model, **data, **conf}
    else:
        export = [{"model":model, "data":data, "predictions":predictions,}]
        config = {**model, **data, **predictions}
    
        
        if args.features:
            config["features"] = args.features
            choices = ["radiomics", "whole_brain","tumor","combined"]
            assert args.features in choices, f"args.features should be in {choices}"
            config["output_name"] = args.features

        if args.output_name:
            config["output_name"] = args.output_name
            

    # exporting config
    with open(join(model["model_path"], f"logs/config_{exp}.yaml"),"w") as file:
        yaml.dump(export, file)

    return config

# %% datasets
def import_datasets(        
         model_path,
         data_path,
         dataset,
         modalities,
         batch_size,
         num_workers,
         **kwargs):

    train_prop = 0.8

    # Data transformers
    # normalTransform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Dataloaders
    print("\nLoading datasets...")
    if exists(model_path + "/exports/trainLoader.pth"):
        print("Restoring previous...")
        trainLoader = torch.load(model_path + "/exports/trainLoader.pth")
        testLoader = torch.load(model_path + "/exports/testLoader.pth")

    else:
        if type(dataset) == list:
            data_path = [join(data_path, z) for z in dataset]
        else:
            data_path = [join(data_path, dataset)]
            
        totalData = BrainImages(data_path, modalities)
        trainData, testData = random_split(
            totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

        fullLoader = DataLoader(
            totalData, batch_size=1, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
        trainLoader = DataLoader(
            trainData, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
        testLoader = DataLoader(
            testData, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)
        
        # Saving datasets
        torch.save(fullLoader, f'{model_path}/exports/fullLoader.pth')
        torch.save(trainLoader, f'{model_path}/exports/trainLoader.pth')
        torch.save(testLoader, f'{model_path}/exports/testLoader.pth')

    return trainLoader, testLoader


# %% Models
def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # step is in [0,1]

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0 + np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L


def import_model(type_ae, modalities, features, num_blocks, min_dims, **config):

    # Initialize a model of our autoEncoder class on the device
    assert type_ae.lower() in ["ae","unet","oldae","oldaeu","u_vae", "vae3d","rnet"]
    attn = config['attention'] 


    if type_ae.lower() in ["ae", "unet"]:
        net = BaseAE(len(modalities), features, num_blocks, type_ae=type_ae, attention=attn)
    elif type_ae.lower() in ["oldae", "oldaeu"]:
        net = OldAE(len(modalities), features, num_blocks, type_ae=type_ae, attention=attn)
    elif type_ae.lower() == "u_vae":
        net = U_VAE(len(modalities),  features,
                    num_blocks, min_dims, attention=attn, hidden_size=2048)
    elif type_ae.lower() in ["vae3d"]:
        if config["latent_dim"]:
            net = VAE3D(in_channels=len(modalities), latent_dim=config["latent_dim"], min_dims=min_dims)
        else:
            net = VAE3D(in_channels=len(modalities), min_dims=min_dims)
    elif type_ae.lower() == "rnet":
        net = RNet(len(modalities))
    else:
        print("Wrong type_ae (should be in ae, unet, oldae, oldaeu, u_vae, vae3d, rnet) : using ae")
        net = BaseAE(len(modalities), features, num_blocks, type_ae=type_ae)

    beta_dict = None
    try:
        beta = config["beta"]
        num_epochs = config['num_epochs'] 

        if beta == "sigmoid":
            beta_dict = {i: frange_cycle_sigmoid(0.0, 0.02, num_epochs, 5)[
                i] for i in range(num_epochs)}
        else:
            beta_dict = {i: float(beta) for i in range(num_epochs)}
    except:
        pass

    return net, beta_dict


def export_model(net, model_path, epoch, test_epoch_metrics, best_test_loss):
    if epoch % 20 == 0:
        torch.save(
            net.state_dict(), model_path +
            f"/exports/model_ep{epoch}.pt")

    if test_epoch_metrics['loss'] < best_test_loss:
        torch.save(
            net.state_dict(),
            model_path+f"/exports/best_model.pt")
        best_test_loss = test_epoch_metrics['loss']

    return best_test_loss


# %% Logs
def update_curves(report,  output_dir):

    print("Updating metric curves...")
    metrics = ["loss", "ssim"]

    fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(6, 8))
    for i, metric in enumerate(metrics):
        ax[i].plot(report.index, report["train_" + metric], label='Train')
        ax[i].plot(report.index, report["test_" + metric], label='Test')

        ax[i].set_title(metric.upper())
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[0].legend()
    fig.savefig(f"{output_dir}/logs/curves.png")
    plt.close(fig)


def update_report(
    config, train_cases, val_cases, optimizer,
    epoch, train_epoch_metrics, test_epoch_metrics, beta
):

    output_dir = config['model_path']

    report_path = f"{output_dir}/logs/report.csv"
    report = pd.read_csv(report_path) if os.path.exists(
        report_path) else pd.DataFrame()

    epoch_report = {}
    epoch_report['model'] = [config['model_name']]
    epoch_report['train cases'] = int(train_cases)
    epoch_report['val cases'] = int(val_cases)
    epoch_report['num_mod'] = len(config['modalities'])
    epoch_report['init_features'] = int(config['features'])
    epoch_report['batch_size'] = config['batch_size']
    epoch_report['learning rate'] = optimizer.param_groups[0]['lr']
    epoch_report['total_epoch'] = config["num_epochs"]
    epoch_report['epoch'] = int(epoch)+1
    epoch_report['train_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_report['train_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_report['test_loss'] = round(test_epoch_metrics['loss'], 6)
    epoch_report['test_ssim'] = round(test_epoch_metrics['ssim'], 6)
    epoch_report['beta'] = round(beta, 3)

    epoch_report = pd.DataFrame.from_dict(epoch_report)
    report = pd.concat([report, epoch_report], ignore_index=True)
    report.to_csv(report_path, index=False)

    return report


def reconstruct_image(net, device, output_dir, testloader, **kwargs):

    imgs, _ = next(iter(testloader))

    batch_size, nmod, depth, height, width = imgs.size()
    orig = torch.reshape(
        imgs[:, :, depth // 2, :, :],
        [batch_size * nmod, 1, height, width]
    )

    net.eval()
    imgs = imgs.to(device)
    reconstructed = net(imgs)[0]
    reconstructed = reconstructed.cpu().data

    reconstructed = torch.reshape(
        reconstructed[:, :, depth // 2, :, :],
        [batch_size*nmod, 1, height, width]
    )

    concat = torch.cat([orig, reconstructed])
    print("conc ", concat.size())
    concat = torch.flip(concat, dims=(2,))

    save_image(
        make_grid(concat, nrow=nmod*batch_size),
        f'{output_dir}/logs/reconstructions.png'
    )

    return
