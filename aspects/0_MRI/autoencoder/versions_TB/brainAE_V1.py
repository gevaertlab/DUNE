# GECO = Gan-Esque Convolutional Obscuration because acronyms

#################
###
# Demander à Bryce comment organiser les données d'entrée.
###
###
###
##################
# %%

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from numpy import array
import math as math
import copy as copy
import torch.optim as optim
import torch.nn.functional as func
import torch.nn as nn
import torch.autograd as autograd
import csv
import random
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
# %matplotlib inline

NMOD = 3
SUBSET = 75

os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
MODALITIES = ["t1", "t2", "t1Gd", "flair", "GlistrBoost"][:NMOD]
DATA = ["./data/data_fusion/MR/dispatch/" + mod for mod in MODALITIES]
OUTPUT = "./data/data_fusion/MR/outputs/"

# The convFilter of GECO, which is intended to filter out device-specific artifacts from images


class convAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Creating the neural network structure
        self.conv_layers = kwargs["conv_layer_count"]
        self.lastOut = []

        # Encoder
        self.conv1 = nn.Conv3d(
            in_channels=NMOD, out_channels=32, kernel_size=[3, 3, 3])
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=12, kernel_size=[3, 3, 3])
        self.conv3 = nn.Conv3d(
            in_channels=12, out_channels=1, kernel_size=[3, 3, 3])
        # self.conv4 = nn.Conv3d(
        # in_channels=8, out_channels=1, kernel_size=[4,4,4])
        self.pool = nn.MaxPool3d(4, 1, return_indices=True)

        # Decoder
        self.unpool = nn.MaxUnpool3d(4, 1)
        # self.t_conv4 = nn.ConvTranspose3d(
        # in_channels=1, out_channels=8, kernel_size=[4,4,4])
        self.t_conv3 = nn.ConvTranspose3d(
            in_channels=1, out_channels=12, kernel_size=[3, 3, 3])
        self.t_conv2 = nn.ConvTranspose3d(
            in_channels=12, out_channels=14, kernel_size=[3, 3, 3])
        self.t_conv1 = nn.ConvTranspose3d(
            in_channels=14, out_channels=NMOD, kernel_size=[3, 3, 3])

    def forward(self, features):
        # Encode
        x = func.relu(self.conv1(features))
        x, id1 = self.pool(x)
        x = func.relu(self.conv2(x))
        x, id2 = self.pool(x)
        # x = func.relu(self.conv3(x))
        # x, id3 = self.pool(x)
        # x = func.relu(self.conv4(x))
        # x, id4 = self.pool(x)

        # print(f"Code shape is {x.shape}.")

        # Decode
        # x = self.unpool(x, id4)
        # x = func.relu(self.t_conv4(x))
        # x = self.unpool(x, id3)
        # x = func.relu(self.t_conv3(x))
        x = self.unpool(x, id2)
        x = func.relu(self.t_conv2(x))
        x = self.unpool(x, id1)
        x = func.relu(self.t_conv1(x))

        return x  # features


def saveDecodedImage(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_img(img, OUTPUT+'ValidationImgs/AEValImage{}.png'.format(epoch))


def find_brain_center(stack):
    """
    Find coordinates of the brain center
    """
    sequence = stack[0]
    D, H, W = sequence.shape
    x_limits = [x for x in range(W) if sequence[:, :, x].sum().item() > 0]
    y_limits = [y for y in range(H) if sequence[:, y, :].sum().item() > 0]
    z_limits = [z for z in range(D) if sequence[z, :, :].sum().item() > 0]

    center = (
        int((z_limits[-1] + z_limits[0])/2),
        int((y_limits[-1] + y_limits[0])/2),
        int((x_limits[-1] + x_limits[0])/2))

    return center


class brainImages(Dataset):
    def __init__(self, minDims, pathsToData=DATA, modalities=MODALITIES, transforms=None):
        self.folders = []
        for i in range(0, len(pathsToData)):
            #print("the "+str(i)+" path to data = "+str(pathsToData[i]))
            self.folders.append(glob.glob(pathsToData[i]+"/*")[:SUBSET])
            self.folders[i] = sorted(self.folders[i])
            # print("the "+str(i)+" folder = "+str(self.folders[i]))
        self.transforms = transforms
        self.modalityCount = len(pathsToData)
        self.modalities = modalities
        self.depth, self.height, self.width = minDims

    def __len__(self):
        #print("there are "+str(len(self.folders[0]))+" image pairs")
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
            try:
                for z in range(startSliceZ, endSliceZ):
                    temp = imgsPre[i, z, startSliceY:endSliceY,
                                   startSliceX:endSliceX].numpy()
                    temp = Image.fromarray(temp)
                    imgsPil[i, z-startSliceZ, :,
                            :] = transforms.ToTensor()(temp)
            except:
                print(f"index error : {idx}")
                print("end slice = ", startSliceZ, z, endSliceZ)

        imgs = imgsPil[:, :, :, :]*255

        return imgs, case


def filterLossFunc(outputs, inputs, biases, detLossScore):
    return biases[0].float()*nn.MSELoss(outputs, inputs) + biases[1].float()*detLossScore


def testImageReconstruction(net, testloader, device, minDims, fileName, slicePad, epoch):
    print("Image reconstruction")
    for batch in testloader:
        img, _ = batch
        #print("img shape = "+str(img.shape))
        #print("minDims = "+str(minDims))
        imgToSaveHeight = torch.reshape(
            img[:, 0, :, int(minDims[1]/2), :], [img.size(0), 1, minDims[0], minDims[2]])
        imgToSaveWidth = torch.reshape(
            img[:, 0, :, :, int(minDims[2]/2)], [img.size(0), 1, minDims[0], minDims[1]])
        imgToSaveDepth = torch.reshape(
            img[:, 0, int(minDims[0]/2), :, :], [img.size(0), 1, minDims[1], minDims[2]])
        #print("imgToSave shape = "+str(imgToSaveHeight.shape))
        save_image(torchvision.utils.make_grid(
            imgToSaveHeight), OUTPUT+"checking_inputsH.png")
        save_image(torchvision.utils.make_grid(
            imgToSaveWidth), OUTPUT+"checking_inputsW.png")
        save_image(torchvision.utils.make_grid(
            imgToSaveDepth), OUTPUT+"checking_inputsD.png")

        # img = img[:, :, int(minDims[0]/2) -
        #   slicePad:int(minDims[0]/2)+slicePad+1, :, :]
        img = img.to(device)
        #img = img.view(img.size(0), -1)

        outputs = net(img)
        outputs.view(outputs.size(0), outputs.size(
            1), minDims[1], minDims[0], minDims[2]).cpu().data
        outputs = outputs.cpu().data
        outputsT1 = torch.reshape(outputs[:, 0, slicePad, :, :], [
            outputs.size(0), 1, minDims[1], minDims[2]])
        outputsT2 = torch.reshape(outputs[:, 1, slicePad, :, :], [
            outputs.size(0), 1, minDims[1], minDims[2]])
        #print("outputsT1 type = "+str(outputsT1.dtype)+" and shape = "+str(outputsT1.shape))
        T1max = torch.max(outputsT1)
        T1min = torch.min(outputsT1)
        T2max = torch.max(outputsT2)
        T2min = torch.min(outputsT2)
        outputsT1 = ((outputsT1-T1min)/(T1max-T1min))

        print(f"saving image in {fileName}")
        save_image(torchvision.utils.make_grid(
            outputsT1), f'{OUTPUT}_epoch{epoch}_T1_reconstruction.png')
        save_image(torchvision.utils.make_grid(
            outputsT2), f'{OUTPUT}_epoch{epoch}_T2_reconstruction.png')
        break


def computeOuterLinDim(batchSize, slicePad, convLayerCount, minImageDims, convChannels, convKernelSizes):
    height = minImageDims[1]
    width = minImageDims[2]
    depth = 2*slicePad+1
    channels = convChannels[0]

    for i in range(0, convLayerCount):
        depth = (depth - convKernelSizes[2])+1
        height = (height - convKernelSizes[0])+1
        width = (width - convKernelSizes[1])+1
        channels = convChannels[i+1]
    return depth*height*width*channels


def fullTrain(testImageFileName=OUTPUT+"check_learning_worked.png", labels=OUTPUT+'ukb_Sex_BirthYear.csv', numEpochs=5, learningRate=1e-4, trainFrac=0.9, runFrac=1, batchSize=5, slicePad=4, codeDimension=5000, convLayerCount=4, minImageDims=[150, 175, 155], convChannels=[len(MODALITIES), 1, 1, 1, 1], convKernelSizes=[3, 3, 3]):

    learningRate = learningRate/(runFrac/0.1)
    # Defining the transforms we want performed on the data
    normalTransform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    preCodeDimension = computeOuterLinDim(
        batchSize, slicePad, convLayerCount, minImageDims, convChannels, convKernelSizes)

    # If slicePad is less than the number of convolutional layers, you get problems. As such, we check for this and warn user.
    if slicePad*convKernelSizes[0] < (convLayerCount+1):
        print("Slice padding around central layer (for 2.5-dimensional image processing) is less than the number of convolutional layers in the encoder. This will cause a crash. Please decrease the number of layers, or increase slice padding value.")

    # Pick a device. If GPU available, use that. Otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize a model of our autoEncoder class on the device
    ae = convAE(conv_layer_count=convLayerCount, conv_channels=convChannels, conv_kernel_sizes=convKernelSizes,
                conv_channels_backwards=convChannels[::-1], conv_kernel_sizes_backwards=convKernelSizes, linear_outer_feat=preCodeDimension, linear_inner_feat=codeDimension).to(device)
    # Define the optimization problem to be solved
    aeOpt = optim.Adam(ae.parameters(), lr=learningRate)
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
        trainData, batch_size=batchSize, shuffle=True, drop_last=True)
    testLoader = DataLoader(
        testData, batch_size=batchSize, shuffle=True, drop_last=True)
    torch.save(trainLoader, 'trainLoader.pth')
    torch.save(testLoader, 'testLoader.pth')

    print("There are "+str(len(trainLoader)*batchSize)+" training samples")
    aeLossTracker = []
    print(len(trainLoader))
    print(len(testLoader))

    # Main loop
    for epoch in range(numEpochs):
        print("Starting epoch "+str(epoch))
        aeLoss = 0
        for batch_features, case in trainLoader:
            print(f'Processing {case}...')

            print("Shape of bactch feature is : ", batch_features.shape)
            aeOpt.zero_grad()
            batch_features = batch_features.to(device)
            aeOutputs = ae(batch_features)

            ae_train_loss = aeCriterion(aeOutputs, batch_features)
            ae_train_loss.backward()
            aeOpt.step()
            aeLoss += ae_train_loss.item()

        # Normalize loss for epoch
        aeLoss = aeLoss/(len(trainLoader)*batchSize*runFrac)
        aeLossTracker.append(aeLoss)
        # Print epoch num and corresponding loss
        print("Epoch: {}/{}, autoencoder loss = {:6f}".format(epoch, numEpochs, aeLoss))

        # Validation
        testImageReconstruction(ae, testLoader, device, minImageDims,
                                OUTPUT+"training_test_epoch_"+str(epoch)+".png", slicePad, epoch)

    plt.figure()
    plt.plot(aeLossTracker)
    plt.title('Training Loss')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.savefig(OUTPUT+'loss.png')

    torch.save(ae.state_dict(), OUTPUT+"brain_ae_model_"+str(preCodeDimension)+"_codeDim_" +
               str(runFrac)+"_runFrac_fullVolEval.pth")

    with open(OUTPUT+'saved_loss_trajectory_'+str(preCodeDimension)+'_codeDim_'+str(runFrac)+'_runFrac.csv', mode='w') as trajectoryFile:
        writer = csv.writer(trajectoryFile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(aeLossTracker)

    testImageReconstruction(ae, testLoader, device,
                            minImageDims, testImageFileName, slicePad, epoch)


# %%
if __name__ == "__main__":
    fullTrain()
    # pass

# %%

# DEBUG
##########

# testImageFileName = OUTPUT+"check_learning_worked.png"
# numEpochs = 1
# learningRate = 1e-4
# trainFrac = 0.9
# runFrac = 1
# batchSize = 5
# slicePad = 4
# codeDimension = 5000
# convLayerCount = 4
# minImageDims = [155, 175, 155]
# convChannels = [len(MODALITIES), 1, 1, 1, 1]
# convKernelSizes = [3, 3, 3]


# learningRate = learningRate/(runFrac/0.1)
# # Defining the transforms we want performed on the data
# normalTransform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# preCodeDimension = computeOuterLinDim(
#     batchSize, slicePad, convLayerCount, minImageDims, convChannels, convKernelSizes)

# # If slicePad is less than the number of convolutional layers, you get problems. As such, we check for this and warn user.
# if slicePad*convKernelSizes[2] < (convLayerCount+1):
#     print("Slice padding around central layer (for 2.5-dimensional image processing) is less than the number of convolutional layers in the encoder. This will cause a crash. Please decrease the number of layers, or increase slice padding value.")

#     # Pick a device. If GPU available, use that. Otherwise, use CPU.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ae = convAE(conv_layer_count=convLayerCount, conv_channels=convChannels, conv_kernel_sizes=convKernelSizes,
#             conv_channels_backwards=convChannels[::-1], conv_kernel_sizes_backwards=convKernelSizes, linear_outer_feat=preCodeDimension, linear_inner_feat=codeDimension).to(device)


# totalData = brainImages(minDims=minImageDims,
#                         transforms=normalTransform, pathsToData=DATA, modalities=MODALITIES)

# trainData, testData = torch.utils.data.dataset.random_split(totalData, [int(
#     len(totalData)*trainFrac), int(len(totalData))-int(len(totalData)*trainFrac)])
# trainFracs = [int(len(trainData)*runFrac),
#               len(trainData)-int(len(trainData)*runFrac)]
# testFracs = [int(len(testData)*runFrac),
#              len(testData)-int(len(testData)*runFrac)]
# trainData, _ = torch.utils.data.dataset.random_split(
#     trainData, trainFracs)
# testData, _ = torch.utils.data.dataset.random_split(
#     testData, testFracs)

# trainLoader = DataLoader(
#     trainData, batch_size=batchSize, shuffle=True, drop_last=True)
# testLoader = DataLoader(
#     testData, batch_size=batchSize, shuffle=True, drop_last=True)


# summary(ae, (3, 155, 175, 155))


# %%

# DISPLAY


# rows = 4; cols = 3;
# fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 16), squeeze=0, sharex=True, sharey=True)
# axes = np.array(axes)

# for i, ax in enumerate(axes.reshape(-1)):
#   ax.set_ylabel(f'Subplot: {i}')
#   ax.imshow(totalData[i][0][1,100])
