"""

NOUVELLE ARCHITECTURE DE LA DATABASE
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
matplotlib.use('Agg')


# from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

###############
### OBJECTS ###
###############


class brainImages(Dataset):
    def __init__(self, pathsToData, transforms=None):
        self.folders = [case for case in os.listdir(
            DATA) if os.path.isdir(os.path.join(DATA, case))]

        self.modalities = MODALITIES
        self.transforms = transforms
        self.n_mod = len(self.modalities)
        self.depth, self.height, self.width = MINDIMS

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        self.imgAddresses = []
        for mod in self.modalities:
            path = glob.glob(DATA+self.folders[idx] + f"/*{mod}.nii*")
            self.imgAddresses.append(path[0])

        case = self.folders[idx]
        imgsPre = []
        for i in range(0, self.n_mod):
            temp = nib.load(self.imgAddresses[i])
            temp = temp.get_fdata().transpose((2, 1, 0))
            temp = np.array(temp)
            temp *= 255.0 / temp.max()
            temp = temp.astype(np.uint8)
            imgsPre.append(temp)

        imgsPre = np.array(imgsPre)
        imgsPre = torch.tensor(imgsPre)

        centerZ, centerY, centerX = find_brain_center(imgsPre)

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
                temp = imgsPre[i, z, startSliceY:endSliceY,
                               startSliceX:endSliceX].numpy()
                temp = Image.fromarray(temp)

                imgsPil[i, z-startSliceZ, :,
                        :] = ToTensor()(temp)

        imgs = imgsPil[:, :, :, :]

        return imgs, case


class convAE(nn.Module):
    def __init__(self, in_c, out_c, num_feat, num_blocks):
        super().__init__()
        # Creating the neural network structure
        self.lastOut = []

        self.num_blocks = num_blocks

        # Encoder tools
        self.conv1 = nn.Conv3d(
            in_channels=in_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.conv2 = nn.Conv3d(
            in_channels=out_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, stride=2, padding=0, return_indices=True)

        # Feature extraction layer
        self.extr = nn.Conv3d(
            in_channels=out_c, out_channels=num_feat, kernel_size=[3, 3, 3], stride=1, padding=1)

        # Decoder tools
        self.unpool = nn.MaxUnpool3d(2, stride=2, padding=0)
        self.start_decode = nn.ConvTranspose3d(
            in_channels=num_feat, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv2 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=in_c, kernel_size=[3, 3, 3], stride=1, padding=1)

    def encoder(self, features):
        indexes, shapes = [], [features.size()]
        for k in range(self.num_blocks):
            if k == 0:
                x = func.relu(self.conv1(features))
            else:
                x = func.relu(self.conv2(x))

            x = func.relu(self.conv2(x))
            x, idk = self.pool(x)
            indexes.append(idk)
            shapes.append(x.size())

        coded_img = self.extr(x)

        return coded_img, indexes, shapes

    def decoder(self, x, indexes, shapes):
        indexes.reverse()
        shapes.reverse()

        x = self.start_decode(x)
        for k in range(self.num_blocks):
            x = self.unpool(x, indexes[k], output_size=shapes[k+1])
            x = func.relu(self.t_conv2(x))

            if k != self.num_blocks-1:
                x = func.relu(self.t_conv2(x))
            else:
                decoded_img = func.relu(self.t_conv1(x))

        decoded_img = decoded_img / torch.max(decoded_img)

        return decoded_img

    def forward(self, features):

        coded_img, indexes, shapes = self.encoder(features)
        decoded_img = self.decoder(coded_img, indexes, shapes)

        return decoded_img

    def get_code_shape(self, features,):
        code, _, _ = self.encoder(features)
        num_features = np.prod(code.shape[1:])

        return code.shape, num_features


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


def dummy_model(features):
    return torch.full(size=features.size(), fill_value=TRAIN_AVERAGE_PIXEL)


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


def evaluate(model, testLoader, criterion, device):
    model.eval()
    loss_list = []
    ssim_list, psnr_list = [], []
    ssim_measurer = SSIM().to(device)
    psnr_measurer = PSNR().to(device)

    for idx, (imgs, caseid) in enumerate(testLoader):
        try:
            batch_images = imgs.to(device)
            with torch.no_grad():
                outputs = model(batch_images)
                loss = criterion(outputs, batch_images)
                # loss = criterion(batch_images, outputs,
                #  data_range=batch_images.max().unsqueeze(0))
                loss_list.append(loss.item())
                # psnr = psnr_measurer(batch_images, outputs)
                # psnr_list.append(psnr.item())
                psnr_list.append(0)
                # ssim = ssim_measurer(batch_images, outputs)
                # ssim_list.append(ssim.item())
                ssim_list.append(0)
        except AssertionError as e:
            print(caseid)
            print(ssim_list)

            print("min images", torch.min(batch_images))
            print("max images", torch.max(batch_images))
            print("min outputs", torch.min(outputs))
            print("max outputs", torch.max(outputs))

            np.save(OUTPUT+"/batch.npy", batch_images.cpu().numpy())
            np.save(OUTPUT+"/output.npy", outputs.cpu().numpy())

            print(e)
            break

    val_loss = np.mean(loss_list)
    val_ssim = np.mean(ssim_list)
    val_psnr = np.mean(psnr_list)

    metric_results = {'loss': val_loss, 'ssim': val_ssim, 'psnr': val_psnr}

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
        reconstructed = net(imgs)
        reconstructed = reconstructed.cpu().data
        reconstructed = torch.reshape(reconstructed[:, :, Z_slice, :, :], [
            batch_size*n_mod, 1, height, width])

        extracted_features, _, _ = net.encoder(imgs)
        extracted_features = extracted_features.cpu().data
        extracted_features = torch.reshape(extracted_features[:, :, extracted_features.size(2)//2, :, :], [
            extracted_features.size(0), 1, extracted_features.size(3), extracted_features.size(4)])
        save_image(torchvision.utils.make_grid(
            extracted_features, nrow=n_mod), f'{output_dir}extracted.png'
        )

        concat = torch.cat([orig, reconstructed])

        save_image(torchvision.utils.make_grid(
            concat, nrow=n_mod), f'{output_dir}concat.png'
        )
        break


def update_curves(train_metrics, test_metrics):

    print("Updating metric curves...")
    metrics = [k for k in train_metrics.keys()]

    fig, ax = plt.subplots(len(metrics), sharex=True, figsize=(7, 14))
    for i, metric in enumerate(metrics):
        ax[i].plot(train_metrics[metric], label='Train')
        ax[i].plot(test_metrics[metric], label='Test')
        ax[i].set_title(metric.upper())
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel(metric)
        ax[0].legend()
    fig.savefig(f"{OUTPUT}/curves.png")


def generate_report(model_name, epoch, train_epoch_metrics, test_epoch_metrics, num_features):

    epoch_summary = {}
    epoch_summary['model'] = model_name
    epoch_summary['run_date'] = date.today().strftime("%Y/%m/%d")
    epoch_summary['quick_exec'] = config["quick"]
    epoch_summary['num_cases'] = int(NUM_CASES)
    epoch_summary['num_mod'] = int(NMOD)
    epoch_summary['modalities'] = MODALITIES
    epoch_summary['batch_size'] = int(BATCH_SIZE)
    epoch_summary['num_blocks'] = NUM_BLOCKS
    epoch_summary['num_nodes'] = NODES
    epoch_summary['learning rate'] = LEARNING_RATE
    epoch_summary['total_epoch'] = NUM_EPOCHS
    epoch_summary['epoch'] = int(epoch)+1
    epoch_summary['train_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_summary['train_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_summary['train_psnr'] = round(train_epoch_metrics['psnr'], 6)
    epoch_summary['test_loss'] = round(train_epoch_metrics['loss'], 6)
    epoch_summary['test_ssim'] = round(train_epoch_metrics['ssim'], 6)
    epoch_summary['test_psnr'] = round(train_epoch_metrics['psnr'], 6)
    epoch_summary['num_features'] = int(num_features)
    epoch_summary['average_brain_loss'] = float(AVERAGE_BRAIN_LOSS)

    epoch_summary = pd.DataFrame(epoch_summary)

    return epoch_summary


def main(train_prop=0.9):

    # Pick a device. If GPU available, use that. Otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model of our autoEncoder class on the device
    ae = convAE(in_c=NMOD, out_c=NODES, num_feat=EXTRACTED_DIM,
                num_blocks=NUM_BLOCKS).to(device)

    # check if model version already exists and loads it
    if os.path.exists(OUTPUT + "/model.pth"):
        print("Loading backup version of the model...")
        ae.load_state_dict(torch.load(OUTPUT + "/model.pth"))

    # Define the optimization problem to be solved
    aeOpt = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    # Define the objective function of the above optimization problem
    criterion = nn.MSELoss()
    # criterion = SSIMLoss(spatial_dims=3).to(device)

    # Defining the transforms we want performed on the data
    normalTransform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

    # Loaders for the training and test datasets
    totalData = brainImages(transforms=normalTransform, pathsToData=DATA)
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
    train_metrics = {"loss": [], "ssim": [], "psnr": []}
    test_metrics = {"loss": [], "ssim": [], "psnr": []}
    report = pd.DataFrame()

    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{NUM_EPOCHS}")
        ae.train()

        for idx, (images, cases_id) in enumerate(trainLoader):
            aeOpt.zero_grad()
            images = images.to(device)
            outputs = ae(images)
            if (idx == 0) and (epoch == 0):
                code_shape, num_features = ae.get_code_shape(images)
                print(
                    f"Coded shape is {code_shape}, with a total of {num_features} extracted features.")
            ae_batch_loss = criterion(outputs, images)
            # ae_batch_loss = criterion(
            #     images, outputs, data_range=images.max().unsqueeze(0))
            # print(ae_batch_loss)
            ae_batch_loss.backward()
            aeOpt.step()

            if config['quick'] and idx == 3:
                break

        # Evaluate model after each epoch
        metrics = ['loss', 'ssim', 'psnr']
        train_epoch_metrics = evaluate(
            ae, trainLoader, criterion=criterion, device=device)
        test_epoch_metrics = evaluate(
            ae, testLoader, criterion=criterion, device=device)
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
        model_name = f"V9_{RUN}"
        epoch_summary = generate_report(
            model_name, epoch, train_epoch_metrics, test_epoch_metrics, num_features)
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
    MODALITIES = config['modalities']
    NMOD = len(MODALITIES)
    EXTRACTED_DIM = config['extracted_features_dimension']
    NODES = config['num_nodes']
    NUM_BLOCKS = config['num_blocks']
    BATCH_SIZE = config['batch_size']
    MINDIMS = config['min_dims']
    NUM_EPOCHS = config['num_epochs'] if not config["quick"] else 3
    LEARNING_RATE = config['learning_rate']
    NUM_WORKERS = config['num_workers']
    NUM_CASES = config['n']
    TRAIN_AVERAGE_PIXEL = 0.16312200032375954
    AVERAGE_BRAIN_LOSS = 0.08607663757897713

    ### PATHS ###
    DATA = "./data/data_fusion/MR/UKBIO/"
    if config["quick"]:
        RUN = f"Test_Batches{BATCH_SIZE}_Blocks{NUM_BLOCKS}_Nodes{NODES}"
    else:
        RUN = f"SSIM_Batches{BATCH_SIZE}_Blocks{NUM_BLOCKS}_Nodes{NODES}"

    OUTPUT = os.path.join(config['output_dir'], RUN)
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    os.makedirs(OUTPUT, exist_ok=True)

    # RUN
    print(f"Batches = {BATCH_SIZE}, Blocks = {NUM_BLOCKS}, Nodes = {NODES}.")
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
# BATCH_SIZE = 6
# NUM_WORKERS = 10
# MODALITIES = [
#         "T1Gd",
#         "FLAIR"
#     ]
# DATA = "./data/data_fusion/MR/UKBIO/"
# totalData = brainImages(transforms=normalTransform, pathsToData=DATA)

# # Pick a device. If GPU available, use that. Otherwise, use CPU.
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# # Initialize a model of our autoEncoder class on the device
# ae = convAE(in_c=2, out_c=16, num_feat=1,
#             num_blocks=3).to(device)


# # Defining the transforms we want performed on the data
# normalTransform = transforms.Compose(
#     [ToTensor(), transforms.Normalize((0.5, 0.5,), (0.5, 0.5,))])

# # Loaders for the training and test datasets
# totalData = brainImages(transforms=normalTransform, pathsToData=DATA)
# trainData, testData = random_split(
#     totalData, [int(len(totalData)*train_prop), len(totalData)-int(len(totalData)*train_prop)])

# trainLoader = DataLoader(
#     trainData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
# testLoader = DataLoader(
#     testData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
# print(
#     f"There are {len(trainLoader)*BATCH_SIZE} training samples and {len(testLoader)*BATCH_SIZE} test samples.")

# for idx, (img, caseid) in enumerate(trainLoader):
#     img = img.to(device)
#     outputs = ae(img)
#     break

# crit = SSIMLoss(spatial_dims=3).to(device)
# crit(img, outputs, data_shape = img.max())

# %%
