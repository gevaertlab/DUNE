# %%

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from numpy import array
import torch.nn.functional as func
import torch.nn as nn
import random
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib

# %matplotlib inline

os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")


# PARAMETERS
NMOD = 5
NODES = 16
EXTRACTED_DIM = 1
NUM_BLOCKS = 4
SUBSET = 168  # total 168
BATCH_SIZE = 2
MINDIMS = [145, 175, 148]
NB_EPOCHS = 10


# PATHS
MODALITIES = ["t1", "t2", "flair", "t1Gd",  "GlistrBoost"][:NMOD]
DATA = ["./data/data_fusion/MR/dispatch/" + mod for mod in MODALITIES]
RUN = f"run_{NUM_BLOCKS}_blocks-{NODES}_nodes"
OUTPUT = f"./data/data_fusion/MR/outputs/{RUN}/"

# The convFilter of GECO, which is intended to filter out device-specific artifacts from images


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
        self.pool = nn.MaxPool3d(3, stride=1, padding=0, return_indices=True)

        # Feature extraction layer
        self.extr = nn.Conv3d(
            in_channels=out_c, out_channels=num_feat, kernel_size=[3, 3, 3], stride=1, padding=1)

        # Decoder tools
        self.unpool = nn.MaxUnpool3d(3, stride=1, padding=0)
        self.start_decode = nn.ConvTranspose3d(
            in_channels=num_feat, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv2 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=out_c, kernel_size=[3, 3, 3], stride=1, padding=1)
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels=out_c, out_channels=in_c, kernel_size=[3, 3, 3], stride=1, padding=1)

    def encoder(self, features):

        indexes = []
        for k in range(self.num_blocks):
            if k == 0:
                x = func.relu(self.conv1(features))
            else:
                x = func.relu(self.conv2(x))

            x = func.relu(self.conv2(x))
            x, idk = self.pool(x)
            indexes.append(idk)

        coded_img = self.extr(x)
        # print(f"Coded shape is {coded_img.shape}")

        return coded_img, indexes

    def decoder(self, x, indexes):
        indexes.reverse()

        x = self.start_decode(x)

        for k in range(self.num_blocks):
            x = self.unpool(x, indexes[k])
            x = func.relu(self.t_conv2(x))

            if k != self.num_blocks-1:
                x = func.relu(self.t_conv2(x))
            else:
                decoded_img = func.relu(self.t_conv1(x))

        return decoded_img

    def forward(self, features):

        coded_img, indexes = self.encoder(features)
        decoded_img = self.decoder(coded_img, indexes)

        return decoded_img  # features


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


class brainImages(Dataset):
    def __init__(self, minDims, pathsToData=DATA, modalities=MODALITIES, transforms=None):
        self.folders = []
        for i in range(0, len(pathsToData)):
            # print("the "+str(i)+" path to data = "+str(pathsToData[i]))
            self.folders.append(glob.glob(pathsToData[i]+"/*")[:SUBSET])
            self.folders[i] = sorted(self.folders[i])
            # print("the "+str(i)+" folder = "+str(self.folders[i]))
        self.transforms = transforms
        self.modalityCount = len(pathsToData)
        self.modalities = modalities
        self.depth, self.height, self.width = minDims

    def __len__(self):
        # print("there are "+str(len(self.folders[0]))+" image pairs")
        return len(self.folders[0])

    def __getitem__(self, idx):
        self.imgAddresses = []
        for i in range(0, self.modalityCount):
            self.imgAddresses.append(
                # self.folders[i][idx]+"/"+self.modalities[i]+"/"+self.modalities[i]+".nii")
                self.folders[i][idx])

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


def evaluate(model, testLoader, criterion, device):
    model.eval()
    loss_list = []

    for hemispheras, _ in testLoader:
        for laterality in ["right", "left"]:
            images = hemispheras[laterality]
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, images)

            loss_list.append(loss.item())

    val_loss = np.mean(loss_list)/BATCH_SIZE

    return val_loss


def testImageReconstruction(net, testloader, device, minDims, epoch, Z_slice=75):
    net.eval()
    output_dir = OUTPUT + "reconstructions/"
    depth = minDims[0]
    height = minDims[1]
    width = minDims[2]//2
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
    plt.savefig(f"{OUTPUT}loss.png")


def fullTrain(numEpochs=NB_EPOCHS, learningRate=1e-4, trainFrac=0.9, runFrac=1, batchSize=BATCH_SIZE, slicePad=4, codeDimension=5000, convLayerCount=4, minImageDims=MINDIMS, convChannels=[len(MODALITIES), 1, 1, 1, 1], convKernelSizes=[3, 3, 3]):

    learningRate = learningRate/(runFrac/0.1)

    # Defining the transforms we want performed on the data
    normalTransform = transforms.Compose(
        [ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Pick a device. If GPU available, use that. Otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a model of our autoEncoder class on the device
    ae = convAE(in_c=NMOD, out_c=NODES, num_feat=EXTRACTED_DIM,
                num_blocks=NUM_BLOCKS).to(device)

    # Define the optimization problem to be solved
    aeOpt = torch.optim.Adam(ae.parameters(), lr=learningRate)
    # Define the objective function of the above optimization problem
    aeCriterion = nn.MSELoss()

    # Loaders for the training and test datasets
    trainLoader = None
    testLoader = None

    totalData = brainImages(minDims=minImageDims,
                            transforms=normalTransform, pathsToData=DATA, modalities=MODALITIES)
    trainData, testData = torch.utils.data.dataset.random_split(totalData, [int(
        len(totalData)*trainFrac), int(len(totalData))-int(len(totalData)*trainFrac)])
    trainFracs = [int(len(trainData)*runFrac),
                  len(trainData)-int(len(trainData)*runFrac)]
    testFracs = [int(len(testData)*runFrac),
                 len(testData)-int(len(testData)*runFrac)]
    trainData, _ = torch.utils.data.dataset.random_split(
        trainData, trainFracs)
    testData, _ = torch.utils.data.dataset.random_split(
        testData, testFracs)

    trainLoader = DataLoader(
        trainData, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=5)
    testLoader = DataLoader(
        testData, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=5)

    print(
        f"There are {len(trainLoader)*batchSize} training samples and {len(testLoader)*batchSize} test samples.")

    # Train loop
    train_loss_tracker = []
    test_loss_tracker = []

    for epoch in range(numEpochs):
        print(f"\nStarting epoch {epoch+1}/{numEpochs}")
        ae.train()
        epoch_loss = 0

        for hemispheras, cases_id in trainLoader:
            for laterality in ["right", "left"]:
                images = hemispheras[laterality]
                aeOpt.zero_grad()
                images = images.to(device)
                aeOutputs = ae(images)
                ae_batch_loss = aeCriterion(aeOutputs, images)
                ae_batch_loss.backward()
                aeOpt.step()
                epoch_loss += ae_batch_loss.item()

        epoch_loss = epoch_loss/(len(trainLoader) * batchSize)

        # Evaluate model at each loop
        train_loss = evaluate(
            ae, trainLoader, criterion=aeCriterion, device=device)
        train_loss_tracker.append(train_loss)
        test_loss = evaluate(
            ae, testLoader, criterion=aeCriterion, device=device)
        test_loss_tracker.append(test_loss)

        # Print epoch num and corresponding loss
        print(f"Autoencoder train loss = {train_loss:6f}")
        print(f"Autoencoder test loss = {test_loss:6f}")
        torch.save(ae.state_dict(), OUTPUT+"model.pth")

    plot_losses(train_loss_tracker, test_loss_tracker)

    testImageReconstruction(ae, testLoader, device,
                            minImageDims, epoch, Z_slice=75)


# %%
if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    fullTrain()
