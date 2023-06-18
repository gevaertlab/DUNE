import wandb
from datetime import datetime as dt
from tqdm import tqdm
from os.path import exists
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torchvision import transforms as transforms
from monai.losses.ssim_loss import SSIMLoss

from utils_ae import *


matplotlib.use('Agg')


def train_loop(model, bet, dataloader, optimizer, device, epoch, train, **config):

    model_name = config["model_name"]
    quick = config["quick"]
    num_epochs = config["num_epochs"]

    ssim_func = SSIMLoss(spatial_dims=3).to(device)
    loss_list, ssim_list, bottleneck = [], [], torch.tensor(0)

    log, colour = (f"Model {model_name} - ep {epoch+1}/{num_epochs} - Train",
                   "yellow") if train else (f"Model {model_name} - ep {epoch+1}/{num_epochs} - Val", "blue")
    model.train() if train else model.eval()

    with torch.set_grad_enabled(train):
        for idx, batch in enumerate(tqdm(dataloader, desc=log, colour=colour)):
            imgs, _ = batch

            images = imgs.to(device)
            reconst, bottleneck, distrib = model(images)

            if model.module._type in ["VAE", "U_VAE"]:
                mu, sigma = distrib
                kld = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            else:
                kld = 0

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

            if quick and idx == 5:
                break

    val_loss = np.mean(loss_list)
    val_ssim = np.mean(ssim_list)
    num_features = np.prod(bottleneck.shape[1:])

    metric_results = {'loss': val_loss,
                      'ssim': val_ssim, "num_features": num_features}

    return metric_results


def main(config):

    quick = config['quick']
    config['num_epochs'] = 10 if quick else config['num_epochs']
    model_path = config['model_path']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']

    # Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_test_loss = np.inf

    net, beta_dict = import_model(**config, device=device)

    

    optimizer = Adam(net.parameters(), lr=config['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, verbose=True)

    # Allocate model on several GPUs
    net = nn.DataParallel(net)
    net = net.to(device)
    if exists(model_path + "/exports/best_model.pt"):
        net.load_state_dict(
            torch.load(model_path + "/exports/best_model.pt"))


    trainLoader, testLoader = import_datasets(**config)
    train_cases, val_cases = len(trainLoader) * \
        batch_size, len(testLoader)*batch_size

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

        # wandb.log(test_epoch_metrics)


if __name__ == "__main__":

    config = parse_arguments("ae")
    now = dt.now().strftime("%m%d_%H%M%S")
    # wandb.init(project="Brain_VAE", id=f"{config['model_name']}_{now}", config=config)
    main(config)
    # wandb.finish()
