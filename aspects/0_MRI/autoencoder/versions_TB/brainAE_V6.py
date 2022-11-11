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


###############
### OBJECTS ###
###############

class brainImages(Dataset):
    def __init__(self, minDims, pathsToData, modalities, transforms=None):
        self.folders = []
        for i in range(0, len(pathsToData)):
            # print("the "+str(i)+" path to data = "+str(pathsToData[i]))
            self.folders.append(glob.glob(pathsToData[i]+"/*")[:NUM_CASES])
            self.folders[i] = sorted(self.folders[i])
            # print("the "+str(i)+" folder = "+str(self.folders[i]))
        self.transforms = transforms
        self.modalityCount = len(pathsToData)
        self.modalities = modalities
        self.depth, self.height, self.width = minDims

    def __len__(self):
        return len(self.folders[0])

    def __getitem__(self, idx):
        self.imgAddresses = []
        for i in range(0, self.modalityCount):
            self.imgAddresses.append(self.folders[i][idx])

        case = os.path.basename(self.imgAddresses[0])[:12]
        imgsPre = []
        for i in range(0, self.modalityCount):
            temp = nib.load(self.imgAddresses[i])
            temp = temp.get_fdata().transpose((2, 1, 0))
            temp = np.array(temp)
            temp = temp*255
            temp = temp.astype(np.uint8)
            imgsPre.append(temp)

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
            [self.modalityCount, self.depth, self.height, self.width])

        for i in range(0, self.modalityCount):
            for z in range(startSliceZ, endSliceZ):
                temp = imgsPre[i, z, startSliceY:endSliceY,
                               startSliceX:endSliceX].numpy()
                temp = Image.fromarray(temp)

                imgsPil[i, z-startSliceZ, :,
                        :] = ToTensor()(temp)

        imgs = imgsPil[:, :, :, :]*255

        right_hemi = imgsPil[:, :, :, 0:self.width//2]
        left_hemi = imgsPil[:, :, :, self.width//2:self.width]

        hemispheras = {
            "whole": imgs,
            "right": right_hemi,
            "left": left_hemi
        }

        return hemispheras, case


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

        return decoded_img

    def forward(self, features):

        coded_img, indexes, shapes = self.encoder(features)
        decoded_img = self.decoder(coded_img, indexes, shapes)

        return decoded_img  # features

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

    for hemispheras, _ in testLoader:
        loss_by_case = 0
        for laterality in ["right", "left"]:
            images = hemispheras[laterality]
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, images)

            loss_by_case += loss.item()

        loss_list.append(loss_by_case/2)  # mean loss for the whole brain

    val_loss = np.mean(loss_list)/BATCH_SIZE

    return val_loss


def testImageReconstruction(net, testloader, device, Z_slice=75):
    net.eval()
    output_dir = OUTPUT + "/reconstructions/"
    os.makedirs(output_dir, exist_ok=True)
    depth = MINDIMS[0]
    height = MINDIMS[1]
    width = MINDIMS[2]//2
    batch_size = BATCH_SIZE
    n_mod = NMOD

    for idx, (hemispheras, caseid) in enumerate(testloader):
        for laterality in ["left", "right"]:
            img = hemispheras[laterality]

            orig = torch.reshape(
                img[:, :, Z_slice, :, :], [batch_size*n_mod, 1, height, width])

            save_image(torchvision.utils.make_grid(
                orig, nrow=n_mod), f"{output_dir}{idx}_{laterality}_orig.png")

            img = img.to(device)
            outputs = net(img)
            outputs = outputs.cpu().data

            outputs = torch.reshape(outputs[:, :, Z_slice, :, :], [
                batch_size*n_mod, 1, height, width])

            T1max = torch.max(outputs)
            T1min = torch.min(outputs)
            # outputs = ((outputs-T1min)/(T1max-T1min))

            save_image(torchvision.utils.make_grid(
                outputs, nrow=n_mod), f'{output_dir}{idx}_{laterality}_reconstr.png')
        break


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Test')
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"{OUTPUT}/loss.png")


def generate_report(model_name, epoch, train_loss, test_loss, num_features):

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
    epoch_summary['train_loss'] = round(train_loss, 6)
    epoch_summary['test_loss'] = round(test_loss, 6)
    epoch_summary['num_features'] = int(num_features) * 2  # two hemispheras
    epoch_summary['average_brain_loss'] = float(AVERAGE_BRAIN_LOSS)

    return epoch_summary


def main(train_prop=0.9):

    # Pick a device. If GPU available, use that. Otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model of our autoEncoder class on the device
    ae = convAE(in_c=NMOD, out_c=NODES, num_feat=EXTRACTED_DIM,
                num_blocks=NUM_BLOCKS).to(device)

    # Define the optimization problem to be solved
    aeOpt = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE)
    # Define the objective function of the above optimization problem
    aeCriterion = nn.MSELoss()

    # Defining the transforms we want performed on the data
    normalTransform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loaders for the training and test datasets
    totalData = brainImages(minDims=MINDIMS,
                            transforms=normalTransform, pathsToData=DATA, modalities=MODALITIES)
    trainData, testData = random_split(
        totalData, [int(NUM_CASES*train_prop), int(NUM_CASES)-int(NUM_CASES*train_prop)])

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
    test_loss_tracker = []
    report = pd.DataFrame()

    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting epoch {epoch+1}/{NUM_EPOCHS}")
        ae.train()

        for idx, (hemispheras, cases_id) in enumerate(trainLoader):
            for laterality in ["right", "left"]:
                images = hemispheras[laterality]
                aeOpt.zero_grad()
                images = images.to(device)
                aeOutputs = ae(images)
                if (idx == 0) and (epoch == 0) and (laterality == "right"):
                    code_shape, num_features = ae.get_code_shape(images)
                    print(
                        f"Coded shape is {code_shape}, with a total of {num_features} extracted features.")
                ae_batch_loss = aeCriterion(aeOutputs, images)
                ae_batch_loss.backward()
                aeOpt.step()

        # Evaluate model after each epoch
        train_loss = evaluate(
            ae, trainLoader, criterion=aeCriterion, device=device)
        train_loss_tracker.append(train_loss)
        test_loss = evaluate(
            ae, testLoader, criterion=aeCriterion, device=device)
        test_loss_tracker.append(test_loss)

        # Print epoch num and corresponding loss
        print(f"Autoencoder train loss = {train_loss:6f}")
        print(f"Autoencoder test loss = {test_loss:6f}")
        torch.save(ae.state_dict(), OUTPUT+"/model.pth")

        # Report generation
        model_name = f"new_Batch{BATCH_SIZE}_Blocks{NUM_BLOCKS}_Nodes{NODES}"
        epoch_summary = generate_report(
            model_name, epoch, train_loss, test_loss, num_features)
        report = report.append(epoch_summary, ignore_index=True)
        report.to_csv(f"logs/{model_name}.csv", index=False)

    plot_losses(train_loss_tracker, test_loss_tracker)
    testImageReconstruction(ae, testLoader, device, Z_slice=75)


# %%
############
### MAIN ###
############
if __name__ == "__main__":
    start = datetime.now()

    ### PARAMETERS ###
    config = parse_arguments()
    NMOD = config['num_modalities']
    EXTRACTED_DIM = config['extracted_features_dimension']
    NODES = config['num_nodes']
    NUM_BLOCKS = config['num_blocks']
    BATCH_SIZE = config['batch_size']
    MINDIMS = config['min_dims']
    NUM_EPOCHS = config['num_epochs'] if not config["quick"] else 2
    LEARNING_RATE = config['learning_rate']
    NUM_WORKERS = config['num_workers']
    NUM_CASES = config['n'] if not config["quick"] else 8*BATCH_SIZE
    TRAIN_AVERAGE_PIXEL = 0.16312200032375954
    AVERAGE_BRAIN_LOSS = 0.08607663757897713

    ### PATHS ###
    os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
    MODALITIES = ["t1", "t2", "flair", "t1Gd",  "GlistrBoost"][:NMOD]
    DATA = ["./data/data_fusion/MR/dispatch/" + mod for mod in MODALITIES]
    RUN = f"run_Batches{BATCH_SIZE}_Blocks{NUM_BLOCKS}_Nodes{NODES}"
    OUTPUT = os.path.join(config['output_dir'], RUN)
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
#         loss_dummy = aeCriterion(output_dummy, img)
#         dummy_losses.append(loss_dummy.item())

# np.mean(dummy_losses)
