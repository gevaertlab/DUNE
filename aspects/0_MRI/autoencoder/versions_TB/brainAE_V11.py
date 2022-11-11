"""

NOUVELLE ARCHITECTURE DE LA DATABASE

Peut prendre en input à la fois base TCGA et base UKbiobank
Laisse le choix du criterion
Laisse le choix du nombre de canaux.


"""


# %%
import argparse
import glob
import json
import os
from datetime import datetime, date
import humanfriendly
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
from numpy import array
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import pandas as pd
from piqa import PSNR, SSIM
from monai.losses.ssim_loss import SSIMLoss
from models import UNet, convAE
matplotlib.use('Agg')



###############
### OBJECTS ###
###############


class brainImages(Dataset):
    def __init__(self, data_path, transforms=None):

        self.data_path = data_path
        self.folders = [case for case in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, case))]

        self.modalities = MODALITIES
        self.transforms = transforms
        self.n_mod = len(self.modalities)
        self.depth, self.height, self.width = MINDIMS
        self.dataset = DATASET

    @staticmethod
    def find_brain_center(stack):
        """
        Find coordinates of the brain center
        """
        sequence = stack[0]
        D, H, W = sequence.shape
        z_limits = [z for z in range(D) if sequence[z, :, :].sum().item() > 0]
        y_limits = [y for y in range(H) if sequence[:, y, :].sum().item() > 0]
        x_limits = [x for x in range(W) if sequence[:, :, x].sum().item() > 0]

        center = (
            int((z_limits[-1] + z_limits[0])/2),
            int((y_limits[-1] + y_limits[0])/2),
            int((x_limits[-1] + x_limits[0])/2))

        return center

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        self.imgAddresses = []
        for mod in self.modalities:
            mod = mod.lower() + ".nii"
            folderpath = os.path.join(self.data_path, self.folders[idx])
            filepath = [os.path.join(folderpath, f) for f in os.listdir(
                folderpath) if mod in f.lower()][0]
            self.imgAddresses.append(filepath)

        case = self.folders[idx]
        imgsPre = []
        for i in range(0, self.n_mod):
            temp = nib.load(self.imgAddresses[i])
            temp = temp.get_fdata().transpose((2, 1, 0))
            temp = np.array(temp)
            temp *= 255.0 / temp.max()
            temp = temp.astype(np.uint8)

            imgsPre.append(temp.copy())

        imgsPre = np.array(imgsPre)
        imgsPre = torch.tensor(imgsPre)

        centerZ, centerY, centerX = brainImages.find_brain_center(imgsPre)

        startSliceZ = int(
            centerZ - self.depth/2) if centerZ > self.depth/2 else 0
        endSliceZ = int(startSliceZ) + self.depth

        startSliceY = int(
            centerY - self.height/2) if centerY > self.height/2 else 0
        endSliceY = int(startSliceY) + self.height

        startSliceX = int(
            centerX - self.width/2)if centerX > self.width/2 else 0
        endSliceX = int(startSliceX) + self.width

        imgsPil = torch.zeros(
            [self.n_mod, self.depth, self.height, self.width])

        for i in range(0, self.n_mod):
            for z in range(startSliceZ, endSliceZ):
                t = imgsPre[i, z, startSliceY:endSliceY,
                            startSliceX:endSliceX].numpy()
                t = Image.fromarray(t)

                if self.dataset == "UKBIOBANK":
                    t = t.rotate(angle=180)

                imgsPil[i, z-startSliceZ, :,
                        :] = ToTensor()(t)

        imgs = imgsPil[:, :, :, :]

        return imgs, case


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.json",
                        help='configuration json file')
    parser.add_argument("--quick", "-q", action='store_true',
                        help='reduced database for quicker execution')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    config["quick"] = args.quick

    return config


def evaluate(model, testLoader, ssim_loss_measurer, mse_loss_measurer,  device):
    model.eval()
    loss_list = []
    ssim_list, psnr_list, mse_list = [], [], []

    psnr_measurer = PSNR().to(device)

    for idx, (imgs, caseid) in enumerate(testLoader):
        try:
            batch_images = imgs.to(device)
            with torch.no_grad():
                outputs, bottleneck = model(batch_images)

                ssim_loss = ssim_loss_measurer(batch_images, outputs,
                                               data_range=batch_images.max()).item()
                ssim = 1 - ssim_loss
                mse = mse_loss_measurer(outputs, batch_images).item()
                psnr = psnr_measurer(batch_images, outputs).item()

                if CRITERION == "SSIM":
                    loss = ssim_loss
                else:
                    loss = mse

                loss_list.append(loss)
                mse_list.append(mse)
                ssim_list.append(ssim)
                psnr_list.append(psnr)

        except AssertionError as e:
            print(caseid)
            print(ssim_list)

            print("min images", torch.min(batch_images))
            print("max images", torch.max(batch_images))
            print("min outputs", torch.min(outputs))
            print("max outputs", torch.max(outputs))

            print(e)
            break

    val_loss = np.mean(loss_list)
    val_ssim = np.mean(ssim_list)
    val_psnr = np.mean(psnr_list)
    val_mse = np.mean(mse_list)

    metric_results = {'loss': val_loss, 'ssim': val_ssim,
                      'psnr': val_psnr, 'mse': val_mse}

    return metric_results


def testImageReconstruction(net, testloader, device, Z_slice):
    net.eval()
    output_dir = OUTPUT + "/reconstructions/"
    os.makedirs(output_dir, exist_ok=True)
    depth = MINDIMS[0]
    height = MINDIMS[1]
    width = MINDIMS[2]
    batch_size = BATCH_SIZE
    n_mod = NMOD

    for idx, (imgs, caseid) in enumerate(testloader):

        orig = torch.reshape(
            imgs[:, :, Z_slice, :, :], [batch_size*n_mod, 1, height, width])

        imgs = imgs.to(device)
        reconstructed, extracted = net(imgs)
        reconstructed = reconstructed.cpu().data
        reconstructed = torch.reshape(reconstructed[:, :, Z_slice, :, :], [
            batch_size*n_mod, 1, height, width])

        concat = torch.cat([orig, reconstructed])

        save_image(torchvision.utils.make_grid(
            concat, nrow=n_mod), f'{output_dir}concat.png'
        )
        break


def update_curves(train_metrics, test_metrics):

    print("Updating metric curves...")
    metrics = [k for k in train_metrics.keys()]

    fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(7, 17))
    for i, metric in enumerate(metrics):
        ax[i].plot(train_metrics[metric], label='Train')
        ax[i].plot(test_metrics[metric], label='Test')

        if metric == "loss":
            title = metric.upper() + f" = {CRITERION}"
            col = "red"
        else:
            title = metric.upper()
            col = "black"

        ax[i].set_title(title, color=col)
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[0].legend()
    fig.savefig(f"{OUTPUT}/curves.png")


def generate_report(totalData, model_name, epoch, train_epoch_metrics, test_epoch_metrics):

    epoch_summary = {}
    epoch_summary['model'] = model_name
    epoch_summary['run_date'] = date.today().strftime("%Y/%m/%d")
    epoch_summary['quick_exec'] = config["quick"]
    epoch_summary['num_cases'] = len(totalData)
    epoch_summary['num_mod'] = int(NMOD)
    epoch_summary['modalities'] = MODALITIES
    epoch_summary['batch_size'] = int(BATCH_SIZE)
    epoch_summary['num_blocks'] = NUM_BLOCKS
    epoch_summary['num_nodes'] = NODES
    epoch_summary['criterion'] = CRITERION
    epoch_summary['learning rate'] = LEARNING_RATE
    epoch_summary['total_epoch'] = NUM_EPOCHS
    epoch_summary['epoch'] = int(epoch)+1
    epoch_summary['train_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_summary['train_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_summary['train_psnr'] = round(train_epoch_metrics['psnr'], 6)
    epoch_summary['test_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_summary['test_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_summary['test_psnr'] = round(train_epoch_metrics['psnr'], 6)

    epoch_summary = pd.DataFrame(epoch_summary)

    return epoch_summary


def main(train_prop=0.9):

    # Pick a device. If GPU available, use that. Otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model of our autoEncoder class on the device
    ae = UNet(in_channels=2, out_channels=2, init_features=32)
    ae = nn.DataParallel(ae)
    ae = ae.to(device)
    if os.path.exists(OUTPUT + "/model.pth"):
        print("Loading backup version of the model...")
        ae.load_state_dict(torch.load(OUTPUT + "/model.pth"))

    # Define the optimization problem to be solved
    aeOpt = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)

    # List of loss functions
    ssim_loss_measurer = SSIMLoss(spatial_dims=3).to(device)
    mse_loss_measurer = nn.MSELoss()

    # Defining the transforms we want performed on the data
    normalTransform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Loaders for the training and test datasets
    totalData = brainImages(data_path=DATA, transforms=normalTransform)
    trainData, testData = random_split(
        totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

    trainLoader = DataLoader(
        trainData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    testLoader = DataLoader(
        testData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    print(
        f"There are {len(trainLoader)*BATCH_SIZE} training samples and {len(testLoader)*BATCH_SIZE} test samples.")

    # Saving datasets
    torch.save(trainLoader, f'{OUTPUT}/trainLoader.pth')
    torch.save(testLoader, f'{OUTPUT}/testLoader.pth')

    # Train loop
    train_loss_tracker = []
    train_metrics = {"loss": [], "ssim": [], "psnr": [], "mse": []}
    test_metrics = {"loss": [], "ssim": [], "psnr": [], "mse": []}
    report = pd.DataFrame()

    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{NUM_EPOCHS}")
        ae.train()

        for idx, (images, cases_id) in enumerate(trainLoader):
            aeOpt.zero_grad()
            images = images.to(device)
            outputs, bottleneck = ae(images)

            if CRITERION == "SSIM":
                criterion = ssim_loss_measurer
                ae_batch_loss = criterion(
                    images, outputs, data_range=images.max().unsqueeze(0))
            else:
                criterion = mse_loss_measurer
                ae_batch_loss = criterion(outputs, images)

            # print(ae_batch_loss)
            ae_batch_loss.backward()
            aeOpt.step()

            if config['quick'] and idx == 2:
                break

        # Evaluate model after each epoch
        metrics = ['loss', 'ssim', 'psnr', 'mse']
        train_epoch_metrics = evaluate(
            ae, trainLoader, ssim_loss_measurer=ssim_loss_measurer, mse_loss_measurer=mse_loss_measurer, device=device)
        test_epoch_metrics = evaluate(
            ae, testLoader, ssim_loss_measurer=ssim_loss_measurer, mse_loss_measurer=mse_loss_measurer, device=device)
        for m in metrics:
            train_metrics[m].append(train_epoch_metrics[m])
            test_metrics[m].append(test_epoch_metrics[m])

        # Print epoch num and corresponding loss
        print(f"Autoencoder train loss = {train_epoch_metrics['loss']:6f}")
        print(f"Autoencoder test loss = {test_epoch_metrics['loss']:6f}")
        torch.save(ae.state_dict(), OUTPUT+"/model.pth")

        # Reconstruction
        testImageReconstruction(ae, testLoader, device,
                                Z_slice=MINDIMS[0]//2)

        # Report generation
        model_name = f"V11_{RUN}"
        epoch_summary = generate_report(totalData,
                                        model_name, epoch, train_epoch_metrics, test_epoch_metrics)
        # report = report.append(epoch_summary, ignore_index=True)
        report = pd.concat([report, epoch_summary], ignore_index=True)
        report.to_csv(f"{OUTPUT}/{model_name}.csv", index=False)

        # Updating curves
        update_curves(train_metrics, test_metrics)


# %%
############
### MAIN ###
############
if __name__ == "__main__":
    start = datetime.now()

    ### PARAMETERS ###
    config = parse_arguments()
    DATASET = config['dataset']
    MODALITIES = config['modalities']
    NMOD = len(MODALITIES)
    BATCH_SIZE = config['batch_size']
    EXTRACTED_DIM = config['extracted_features_dimension']
    NODES = config['num_nodes']
    NUM_BLOCKS = config['num_blocks']
    MINDIMS = config['min_dims']
    NUM_EPOCHS = config['num_epochs'] if not config["quick"] else 3
    LEARNING_RATE = config['learning_rate']
    NUM_WORKERS = config['num_workers']
    CRITERION = config['criterion']
    FLAG = config['flag']

    ### PATHS ###
    DATA = f"./data/data_fusion/MR/{DATASET}/"
    if config["quick"]:
        RUN = f"Test_{DATASET}_{CRITERION}_{FLAG}"
    else:
        RUN = f"{DATASET}_{CRITERION}_{FLAG}"

    OUTPUT = os.path.join(config['output_dir'], RUN)
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    os.makedirs(OUTPUT, exist_ok=True)

    # RUN
    print(f"Dataset = {DATASET}")
    print(f"Modalities = {MODALITIES}")
    print(f"Batches = {BATCH_SIZE}, Blocks = {NUM_BLOCKS}, Nodes = {NODES}.")
    print(f"Criterion = {CRITERION}")
    print(f"Quick execution = {config['quick']}")
    main()

    execution_time = humanfriendly.format_timespan(datetime.now() - start)
    print(f"\nFinished in {execution_time}.")

# %%
#####################
### AVERAGE BRAIN ###
#####################

# def dummy_model(features):
#     return torch.full(size=features.size(), fill_value=0.16312200032375954)

# dummy_losses = []

# for idx, (images, case) in enumerate(trainLoader):
#     for laterality in ["right","left"]:
#         img = images[laterality]
#         output_dummy = dummy_model(img )
#         loss_dummy = criterion(output_dummy, img)
#         dummy_losses.append(loss_dummy.item())

# np.mean(dummy_losses)


# %%


# DEBUG

# os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
# train_prop = .9
# normalTransform = transforms.Compose(
#     [ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# MINDIMS = [
#     145,
#     175,
#     148
# ]
# BATCH_SIZE = 2
# NUM_WORKERS = 4
# MODALITIES = [
#     "t1Gd",
#     "flair",
#     "GlistrBoost"
# ]
# DATA = ["./data/data_fusion/MR/dispatch/" + mod for mod in MODALITIES]
# totalData = brainImages(transforms=normalTransform, data_path=DATA)


# plt.imshow(totalData[99][0][0, 75, :, :], cmap="Greys_r")
# totalData[2][1]
# %%
