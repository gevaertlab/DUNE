"""



"""

import os
from datetime import datetime, date
import humanfriendly
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.utils import save_image
import pandas as pd
from piqa import PSNR
from monai.losses.ssim_loss import SSIMLoss
from models import UNet3D
from datasets import BrainImages
from utils import *
matplotlib.use('Agg')



def evaluate(model, testLoader, criterion_name, ssim_func, mse_func, device):
    model.eval()
    loss_list, ssim_list, psnr_list, mse_list = [], [], [], []

    psnr_measurer = PSNR().to(device)

    for _, (imgs, _) in enumerate(testLoader):
        batch_images = imgs.to(device)
        with torch.no_grad():
            outputs, _ = model(batch_images)

            ssim_loss = ssim_func(batch_images, outputs,
                                  data_range=batch_images.max()).item()
            ssim = 1 - ssim_loss
            mse = mse_func(outputs, batch_images).item()
            psnr = psnr_measurer(batch_images, outputs).item()

            if criterion_name == "SSIM":
                loss = ssim_loss
            else:
                loss = mse

        loss_list.append(loss)
        mse_list.append(mse)
        ssim_list.append(ssim)
        psnr_list.append(psnr)

    val_loss = np.mean(loss_list)
    val_ssim = np.mean(ssim_list)
    val_psnr = np.mean(psnr_list)
    val_mse = np.mean(mse_list)

    metric_results = {'loss': val_loss, 'ssim': val_ssim,
                      'psnr': val_psnr, 'mse': val_mse}

    return metric_results


def reconstruct_image(net, device, output_dir, testloader, **kwargs):

    net.eval()
    for _, (imgs, _) in enumerate(testloader):
        batch_size, nmod, depth, height, width = imgs.size()

        orig = torch.reshape(
            imgs[:, :, depth // 2, :, :], [batch_size * nmod, 1, height, width])

        imgs = imgs.to(device)
        reconstructed, extracted = net(imgs)
        reconstructed = reconstructed.cpu().data
        reconstructed = torch.reshape(reconstructed[:, :, depth // 2, :, :], [
            batch_size*nmod, 1, height, width])

        concat = torch.cat([orig, reconstructed])

        save_image(torchvision.utils.make_grid(
            concat, nrow=nmod), f'{output_dir}/autoencoding/reconstructions.png'
        )
        break


def main(
    data_path, dataset, output_dir, learning_rate, modalities, features, num_blocks, min_dims,  batch_size, criterion_name, num_epochs, num_workers, model_name, quick, train_prop=0.8):

    # PRINT LOG
    print(f"Dataset = {dataset}")
    print(f"Modalities = {modalities}")
    print(f"Batches = {batch_size}")
    print(f"Criterion = {criterion_name}")
    print(f"Quick execution = {quick}")

    # Dependencies
    output_dir = create_dependencies(output_dir, model_name)

    # Pick a device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model of our autoEncoder class on the device
    net = UNet3D(in_channels=len(modalities), out_channels=len(modalities), init_features=features, num_blocks=num_blocks)

    # Allocate model on several GPUs
    net = nn.DataParallel(net)
    net = net.to(device)

    if os.path.exists(output_dir + "/autoencoding/exported_data/model.pth"):
        print("Loading backup version of the model...")
        net.load_state_dict(torch.load(
            output_dir + "/autoencoding/exported_data/model.pth"))

    # Optimizer and losses
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    ssim_func = SSIMLoss(spatial_dims=3).to(device)
    mse_func = nn.MSELoss()

    # Data transformers
    normalTransform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Dataloaders
    data_path = os.path.join(data_path, dataset)
    totalData = BrainImages(dataset, data_path, modalities,
                            min_dims, transforms=normalTransform)
    trainData, testData = random_split(
        totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

    trainLoader = DataLoader(
        trainData, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    testLoader = DataLoader(
        testData, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    print(
        f"There are {len(trainLoader)*batch_size} training samples and {len(testLoader)*batch_size} test samples.")

    # Saving datasets
    torch.save(trainLoader, f'{output_dir}/autoencoding/exported_data/trainLoader.pth')
    torch.save(testLoader, f'{output_dir}/autoencoding/exported_data/testLoader.pth')

    # Train loop
    train_metrics = {"loss": [], "ssim": [], "psnr": [], "mse": []}
    test_metrics = {"loss": [], "ssim": [], "psnr": [], "mse": []}

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        net.train()

        for idx, (images, _) in enumerate(trainLoader):
            optimizer.zero_grad()
            images = images.to(device)
            outputs, bottleneck = net(images)

            if idx == 1:
                print(f"Bottleneck shape is {bottleneck.shape} with a total of {np.prod(bottleneck.shape[1:])} features.")

            if criterion_name == "SSIM":
                criterion = ssim_func
                batch_loss = criterion(
                    images, outputs, data_range=images.max().unsqueeze(0))
            else:
                criterion = mse_func
                batch_loss = criterion(outputs, images)

            # print(batch_loss)
            batch_loss.backward()
            optimizer.step()

            if config['quick'] and idx == 2:
                print("quick exec")
                break

        # Evaluate model after each epoch
        metrics = ['loss', 'ssim', 'psnr', 'mse']
        train_epoch_metrics = evaluate(
            net, trainLoader, criterion_name, ssim_func, mse_func, device)
        test_epoch_metrics = evaluate(
            net, testLoader, criterion_name, ssim_func, mse_func, device)
        for m in metrics:
            train_metrics[m].append(train_epoch_metrics[m])
            test_metrics[m].append(test_epoch_metrics[m])

        # Print epoch num and corresponding loss
        print(f"Autoencoder train loss = {train_epoch_metrics['loss']:6f}")
        print(f"Autoencoder test loss = {test_epoch_metrics['loss']:6f}")
        torch.save(net.state_dict(), output_dir+"/autoencoding/exported_data/model.pth")

        # Export results
        reconstruct_image(net, device, output_dir, testLoader)
        report = update_report(output_dir, model_name, quick, totalData, modalities, features, batch_size,
                               criterion_name, learning_rate, num_epochs, epoch, train_epoch_metrics, test_epoch_metrics)
        update_curves(report, criterion_name, output_dir)


if __name__ == "__main__":
    start = datetime.now()
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    config = parse_arguments()
    main(**config)
    execution_time = humanfriendly.format_timespan(datetime.now() - start)
    print(f"\nFinished in {execution_time}.")


