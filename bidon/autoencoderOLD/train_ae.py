import wandb
from datetime import datetime as dt
from tqdm import tqdm
from os.path import exists, join
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms as transforms
from monai.losses.ssim_loss import SSIMLoss

from utils import *
from datasets import BrainImages

from models.oldmodels import OldAE
from models.classicalAE import BaseAE
from models.resnetAE import RNet
from models.VAEs import U_VAE, VAE3D


matplotlib.use('Agg') 

def import_data(        
         model_path,
         data_path,
         dataset,
         modalities,
         whole_brain,
         min_dims,
         batch_size,
         num_workers,
         **kwargs):

    train_prop = 0.8

    # Data transformers
    normalTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Dataloaders
    print("\nLoading datasets...")
    if exists(model_path + "/autoencoding/exported_data/trainLoader.pth"):
        print("Restoring previous...")
        trainLoader = torch.load(model_path + "/autoencoding/exported_data/trainLoader.pth")
        testLoader = torch.load(model_path + "/autoencoding/exported_data/testLoader.pth")

    else:
        data_path = join(data_path, dataset, "images")
        totalData = BrainImages(dataset, data_path, modalities,
                                min_dims, whole_brain=whole_brain, transforms=normalTransform)
        trainData, testData = random_split(
            totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

        trainLoader = DataLoader(
            trainData, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        testLoader = DataLoader(
            testData, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        
        # Saving datasets
        torch.save(
            trainLoader, f'{model_path}/autoencoding/exported_data/trainLoader.pth')
        torch.save(
            testLoader, f'{model_path}/autoencoding/exported_data/testLoader.pth')

    return trainLoader, testLoader


def import_model(type_ae, modalities, features, num_blocks, min_dims, num_epochs, **config):
    
    net, beta_dict = None, None
    # Initialize a model of our autoEncoder class on the device
    if type_ae.lower() in ["ae", "unet"]:
        net = BaseAE(len(modalities), features, num_blocks, type_ae=type_ae)
    elif type_ae.lower() in ["oldae", "oldaeu"]:
        net = OldAE(len(modalities), features, num_blocks, type_ae=type_ae)
    elif type_ae.lower() == "u_vae":
        net = U_VAE(len(modalities),  features, num_blocks, min_dims, hidden_size=2048)
    elif type_ae.lower() in ["vae3d"]:
        if config["latent_dim"]:
            net = VAE3D(latent_dim = config["latent_dim"], min_dims=min_dims)
        else:
            net = VAE3D(min_dims=min_dims)
    elif type_ae.lower() == "rnet":
        net = RNet(len(modalities))

    else:
        print("AE type should be one of followings : AE, UNet, VAE, RNet")
    try:
        beta = config["beta"]

        if beta=="sigmoid":
            beta_dict = {i:frange_cycle_sigmoid(0.0, 0.5, num_epochs, 5)[i] for i in range(num_epochs)}
        else:
            beta_dict = {i:float(beta) for i in range(num_epochs)}
    except:
        pass

    return net, beta_dict


def train_loop(model, bet, dataloader, optimizer, device, epoch, train, **config):
    
    model_name = config["model_name"]
    quick = config["quick"]
    num_epochs = config["num_epochs"]

    ssim_func = SSIMLoss(spatial_dims=3).to(device)
    loss_list, ssim_list, bottleneck = [], [], torch.tensor(0)

    log, colour = (f"Model {model_name} - ep {epoch+1}/{num_epochs} - Train", "magenta") if train else (f"Model {model_name} - ep {epoch+1}/{num_epochs} - Val", "cyan")
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for idx, batch in enumerate(tqdm(dataloader, desc=log, colour=colour)):
            imgs, _ = batch
            images = imgs.to(device)
            reconst, bottleneck, distrib = model(images)

            if model.module._type == "VAE":
                mu, sigma = distrib
                kld = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            else:
                kld=0

            ssimloss = ssim_func(images, reconst, data_range=images.max())
            loss = ssimloss + bet*kld
            ssim_list.append(1 - ssimloss.item())
            loss_list.append(loss.item())
           
            val_loss = np.mean(loss_list)
            val_ssim = np.mean(ssim_list)
            num_features = np.prod(bottleneck.shape[1:])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if quick and idx == 50:
                break
    
    val_loss = np.mean(loss_list)
    val_ssim = np.mean(ssim_list)
    num_features = np.prod(bottleneck.shape[1:])

    metric_results = {'loss': val_loss, 'ssim': val_ssim, "num_features":num_features}

    return metric_results


def export_model(net, model_path, epoch, test_epoch_metrics, best_test_loss):        
    if epoch % 20 == 0:
        torch.save(
            net.state_dict(), model_path +
            f"/autoencoding/exported_data/model_ep{epoch}.pt")

    if test_epoch_metrics['loss'] < best_test_loss:
        torch.save(
            net.state_dict(),
            model_path+f"/autoencoding/exported_data/best_model.pt")
        best_test_loss = test_epoch_metrics['loss']

    return best_test_loss


def main(config):

    quick = config['quick'] 
    config['num_epochs'] = 30 if quick else config['num_epochs']
    model_path = config['model_path'] 
    batch_size = config['batch_size'] 
    num_epochs = config['num_epochs'] 

    print(f"Total epochs = {num_epochs}")


    # Initialization
    create_dependencies(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_test_loss = np.inf

    net, beta_dict = import_model(**config)

    optimizer = Adam(net.parameters(), lr=config['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True)

    # Allocate model on several GPUs
    net = nn.DataParallel(net)
    net = net.to(device)
    if exists(model_path + "/autoencoding/exported_data/best_model.pt"):
        net.load_state_dict(
            torch.load(model_path + "/autoencoding/exported_data/best_model.pt"))


    trainLoader, testLoader = import_data(**config)
    train_cases, val_cases = len(trainLoader)*batch_size, len(testLoader)*batch_size
    

    for epoch in range(num_epochs):
        
        beta = beta_dict[epoch] if beta_dict else 0

        train_epoch_metrics = train_loop(
            net, beta,  trainLoader, optimizer, device, epoch, train=True, **config)
        test_epoch_metrics = train_loop(
            net, beta, testLoader, optimizer, device, epoch, train=False, **config)
        scheduler.step(test_epoch_metrics['loss'])

        # Export epoch results
        reconstruct_image(net, device, model_path, testLoader)
        report = update_report(
            config, train_cases, val_cases, optimizer, epoch,
            train_epoch_metrics, test_epoch_metrics, beta)
        update_curves(report, model_path)


        best_test_loss = export_model(
            net, model_path, epoch, test_epoch_metrics, best_test_loss)   

           
        wandb.log(test_epoch_metrics)




if __name__ == "__main__":

    config = parse_arguments()
    now = dt.now().strftime("%m%d_%H%M%S")
    wandb.init(project="Brain_VAE", id=f"{config['model_name']}_{now}", config=config)
    main(config)


    wandb.finish()
