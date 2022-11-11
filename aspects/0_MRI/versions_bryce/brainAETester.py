#GECO = Gan-Esque Convolutional Obscuration because acronyms

import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import random

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import copy as copy
import math as math

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image

import brainAE as brainMain


#The convFilter of GECO, which is intended to filter out device-specific artifacts from images
class convAE(nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            #Creating the neural network structure
            self.conv_layers=kwargs["conv_layer_count"]
            self.lastOut = []
            self.ae = nn.ModuleList()
            for i in range(0,self.conv_layers):
                self.ae.append(nn.Conv3d(in_channels = kwargs["conv_channels"][i],out_channels = kwargs["conv_channels"][i+1],kernel_size = kwargs["conv_kernel_sizes"][i]))
            for i in range(0,self.conv_layers):
                self.ae.append(nn.ConvTranspose3d(in_channels = kwargs["conv_channels_backwards"][i],out_channels = kwargs["conv_channels_backwards"][i+1],kernel_size = kwargs["conv_kernel_sizes_backwards"][i]))

        def getCode(self,features):
                for i in range(0,self.conv_layers):
                        features = self.ae[i](features)
                        features = func.relu(features)
                return features

        def forward(self, features):
		#features = features.double()
		#Defining progression of data through network
		#Convolve
		#print("Convolve")
                for i in range(0,self.conv_layers):
		#	print("conv layer = "+str(i))
                        #print("conv features = "+str(features))
                        #print("Encode layer = "+str(i))
                        #print("Features dims = "+str(features.shape))
                        features = self.ae[i](features)
                        features = func.relu(features)
		#Deconvolve
		#print("deconvolve")
                for i in range(self.conv_layers,2*self.conv_layers):
		#	print("deconv layer = "+str(i))
                        #print("deconv features = "+str(features))
                        #print("Decode layer = "+str(i))
                        #print("Features dims = "+str(features.shape))
                        features = self.ae[i](features)
                        features = func.relu(features)
                return features



def saveDecodedImage(img,epoch):
	img = img.view(img.size(0),1,28,28)
	save_img(img,'./ValidationImgs/AEValImage{}.png'.format(epoch))




def fullTest(modelFile = "brain_ae_model.pt", trainFrac = 0.7, runFrac = 1, batchSize = 8, minImageDims = [195,160,150]):
        device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ae = torch.load(modelFile)
        print("Loaded the autoencoder model.")
        #Defining the transforms we want performed on the data
        normalTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        #Loaders for the training and test datasets
        totalData = brainMain.brainImages(minDims = minImageDims,transforms=normalTransform,pathsToData=["../../UKBiobankData/T1/"],modalities=["T1"])
        trainData,testData = torch.utils.data.dataset.random_split(totalData,[int(len(totalData)*trainFrac),int(len(totalData))-int(len(totalData)*trainFrac)])
        trainFracs = [int(len(trainData)*runFrac),len(trainData)-int(len(trainData)*runFrac)]
        testFracs = [int(len(testData)*runFrac),len(testData)-int(len(testData)*runFrac)]
        trainData, _ = torch.utils.data.dataset.random_split(trainData,trainFracs)
        testData, _ = torch.utils.data.dataset.random_split(testData,testFracs)
        print("Partitioned data ID #s.")
        trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle=True,drop_last=True)
        testLoader = DataLoader(testData, batch_size = batchSize, shuffle = True,drop_last=True)
        print("Test DataLoader established.")
        print("Testing autoencoder.")
        brainMain.testImageReconstruction(ae,testLoader,device,minImageDims,"T1_reconstruction.png")
        print("Autoencoder test complete.")

fullTest()

#fullTrain()		
