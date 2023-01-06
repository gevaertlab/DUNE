import logging
import os
from datetime import datetime
import humanfriendly
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.utils import save_image
from tqdm import tqdm
from piqa import PSNR
from monai.losses.ssim_loss import SSIMLoss
from sle.models import UNet3D
from datasets import BrainImages
from utils import *


def train_loop(model, dataloader, optimizer, criterion_name, device, train):

    loss_list, ssim_list, psnr_list, mse_list = [], [], [], []
    ssim_func = SSIMLoss(spatial_dims=3).to(device)
    mse_func = nn.MSELoss()
    psnr_func = PSNR().to(device)

    if train:
        model.train()
        print("Train set...")
        for idx, (imgs, cases_id) in enumerate(tqdm(dataloader, colour="magenta")):
            optimizer.zero_grad()
            images = imgs.to(device)
            outputs, bottleneck = model(images)

            ssim_loss = ssim_func(images, outputs, data_range=images.max())
            ssim = 1 - ssim_loss.item()
            mse = mse_func(outputs, images).item()
            psnr = psnr_func(images, outputs).item()

            if criterion_name == "SSIM":
                loss = ssim_loss
            else:
                loss = mse

            loss.backward()
            optimizer.step()

            if idx == 1:
                logging.info(
                    f"Bottleneck shape is {tuple(bottleneck.shape)} with a total of {np.prod(bottleneck.shape[1:])} features.")


            loss_list.append(loss.item())
            mse_list.append(mse)
            ssim_list.append(ssim)
            psnr_list.append(psnr)

            if config['quick'] and idx == 2:
                logging.info("quick exec")
                break

    else:
        model.eval()
        print("Validation set...")
        for idx, (imgs, cases_id) in enumerate(tqdm(dataloader, colour="cyan")):
            images = imgs.to(device)
            with torch.no_grad():
                outputs, bottleneck = model(images)

                ssim_loss = ssim_func(images, outputs, data_range=images.max())
                ssim = 1 - ssim_loss.item()
                mse = mse_func(outputs, images).item()
                psnr = psnr_func(images, outputs).item()

                if criterion_name == "SSIM":
                    loss = ssim_loss
                else:
                    loss = mse

                loss_list.append(loss.item())
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
            concat, nrow=nmod*batch_size), f'{output_dir}/autoencoding/reconstructions.png'
        )
        break


def main(
        data_path, dataset, output_dir, learning_rate, modalities, features, num_blocks, min_dims,  batch_size, criterion_name, num_epochs, num_workers, model_name, quick, train_prop=0.8):

    # Initialization
    output_dir = create_dependencies(output_dir, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_test_loss = np.inf

    # Logging
    logging.basicConfig(filename=os.path.join(output_dir, "ae.log"),
                        filemode='w', format='%(message)s', level=logging.INFO, force=True)
    logging.info(f"Dataset = {dataset}")
    logging.info(f"Modalities = {modalities}")
    logging.info(f"Batches = {batch_size}")
    logging.info(f"Criterion = {criterion_name}")
    logging.info(f"Quick execution = {quick}")

    # Initialize a model of our autoEncoder class on the device
    net = UNet3D(in_channels=len(modalities), out_channels=len(
        modalities), init_features=features, num_blocks=num_blocks)

    # Allocate model on several GPUs
    net = nn.DataParallel(net)
    net = net.to(device)

    if os.path.exists(output_dir + "/autoencoding/exported_data/model.pt"):
        logging.info("Loading backup version of the model...")
        net.load_state_dict(torch.load(
            output_dir + "/autoencoding/exported_data/model.pt"))

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Data transformers
    normalTransform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Dataloaders
    print("\nLoading datasets...")
    if os.path.exists(output_dir + "/autoencoding/exported_data/trainLoader.pth"):
        print("Restoring previous...")
        trainLoader = torch.load(output_dir + "/autoencoding/exported_data/trainLoader.pth")
        testLoader = torch.load(output_dir + "/autoencoding/exported_data/testLoader.pth")

    else:
        data_path = os.path.join(data_path, dataset)
        totalData = BrainImages(dataset, data_path, modalities,
                                min_dims, transforms=normalTransform)
        trainData, testData = random_split(
            totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

        trainLoader = DataLoader(
            trainData, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
        testLoader = DataLoader(
            testData, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    num_cases = len(trainLoader)*batch_size + len(testLoader)*batch_size
    logging.info(
        f"There are {len(trainLoader)*batch_size} training samples and {len(testLoader)*batch_size} test samples.")

    # Saving datasets
    torch.save(
        trainLoader, f'{output_dir}/autoencoding/exported_data/trainLoader.pth')
    torch.save(
        testLoader, f'{output_dir}/autoencoding/exported_data/testLoader.pth')

    for epoch in range(num_epochs):
        logging.info(f"\nStarting epoch {epoch+1}/{num_epochs}")
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_epoch_metrics = train_loop(
            net, trainLoader, optimizer, criterion_name, device, train=True)
        test_epoch_metrics = train_loop(
            net, testLoader, optimizer, criterion_name, device, train=False)

        # logging.info epoch num and corresponding loss
        logging.info(
            f"Autoencoder train loss = {train_epoch_metrics['loss']:6f}")
        logging.info(
            f"Autoencoder test loss = {test_epoch_metrics['loss']:6f}")
        torch.save(net.state_dict(), output_dir +
                   "/autoencoding/exported_data/model.pt")

        if test_epoch_metrics['loss'] < best_test_loss:
            torch.save(net.state_dict(), output_dir+f"/autoencoding/exported_data/best_model.pt")
            best_test_loss = test_epoch_metrics['loss']

        # Export epoch results
        reconstruct_image(net, device, output_dir, testLoader)
        report = update_report(output_dir, model_name, quick, num_cases, modalities, features, batch_size,
                               criterion_name, learning_rate, num_epochs, epoch, train_epoch_metrics, test_epoch_metrics)
        update_curves(report, criterion_name, output_dir)




if __name__ == "__main__":
    start = datetime.now()
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    config = parse_arguments()

    main(**config)
    execution_time = humanfriendly.format_timespan(datetime.now() - start)
    logging.info(f"\nFinished in {execution_time}.")
    os.system('cls')
