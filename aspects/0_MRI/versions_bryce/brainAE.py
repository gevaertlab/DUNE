#GECO = Gan-Esque Convolutional Obscuration because acronyms
# %%
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
import csv


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
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image


#The convFilter of GECO, which is intended to filter out device-specific artifacts from images
class convAE(nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            #Creating the neural network structure
            self.conv_layers=kwargs["conv_layer_count"]
            self.lastOut = []
            self.ae = nn.ModuleList()
            for i in range(0,self.conv_layers):
                self.ae.append(nn.Conv3d(in_channels = kwargs["conv_channels"][i],out_channels = kwargs["conv_channels"][i+1],kernel_size = kwargs["conv_kernel_sizes"]))
            #self.ae.append(nn.Linear(in_features = kwargs["linear_outer_feat"],out_features = kwargs["linear_inner_feat"]))
            #self.ae.append(nn.Linear(in_features = kwargs["linear_inner_feat"],out_features = kwargs["linear_outer_feat"]))
            for i in range(0,self.conv_layers):
                self.ae.append(nn.ConvTranspose3d(in_channels = kwargs["conv_channels_backwards"][i],out_channels = kwargs["conv_channels_backwards"][i+1],kernel_size = kwargs["conv_kernel_sizes_backwards"]))
            print("linear outer dimension = "+str(kwargs["linear_outer_feat"]))


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
                #Inner encoding layer linearizes
                #featuresShape = features.shape
                #features = torch.reshape(features,[featuresShape[0],-1])
                
                

                #print("features length = "+str(features.shape[1]))
                #features = self.ae[self.conv_layers](features)
                #features = func.relu(features)
                #features = self.ae[self.conv_layers+1](features)
                #features = func.relu(features)
                #features = torch.reshape(features,featuresShape)
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


class brainImages(Dataset):
    def __init__(self,minDims,labelFile,pathsToData=["../../UKBiobankData/T1/","../../UKBiobankData/T2/"],modalities=["T1","T2_FLAIR"],layersToInclude=5,transforms=None):
        self.folders = []
        for i in range(0,len(pathsToData)):
            #print("the "+str(i)+" path to data = "+str(pathsToData[i]))
            self.folders.append(glob.glob(pathsToData[i]+"*"))
            self.folders[i] = sorted(self.folders[i])
            #print("the "+str(i)+" folder = "+str(self.folders[i]))
        self.transforms = transforms
        self.modalityCount = len(pathsToData)
        self.modalities = modalities
        self.minimumDims = minDims
        self.savedATest = False
        self.labelData = labelFile
        self.labelsOn = True


    def __len__(self):
        #print("there are "+str(len(self.folders[0]))+" image pairs")
        return len(self.folders[0])

    def includeLabels(self):
        self.labelsOn = True

    def excludeLabels(self):
        self.labelsOn = False

    def __getitem__(self,idx):
        imgsAddresses=[]
        for i in range(0,self.modalityCount):
            imgsAddresses.append(self.folders[i][idx]+"/"+self.modalities[i]+"/"+self.modalities[i]+".nii")
        imgsPre = []
        for i in range(0,self.modalityCount):
            temp = nib.load(imgsAddresses[i])
            temp = temp.get_fdata()
            temp = np.array(temp)
            temp = temp*255
            temp = temp.astype(np.uint8)
            #print("Temp dims = " + str(temp.shape))
            #if self.savedATest == False:
            #    tempToSaveH = torch.tensor(temp)[int(temp.shape[0]/2),:,:]
            #    tempToSaveW = torch.tensor(temp)[:,int(temp.shape[1]/2),:]
            ##    tempToSaveD = torch.tensor(temp)[:,:,int(temp.shape[2]/2)]
            #   tempToSaveH = torch.reshape(tempToSaveH,[1,temp.shape[1],temp.shape[2]])
            #    tempToSaveW = torch.reshape(tempToSaveW,[1,temp.shape[0],temp.shape[2]])
            ##    tempToSaveD = torch.reshape(tempToSaveD,[1,temp.shape[0],temp.shape[1]])
            #    print(str(tempToSaveH.shape))
            #    print(str(tempToSaveW.shape))
            ##    print(str(tempToSaveD.shape))
            #    save_image(tempToSaveH/255,"temp_savedH.png")
            #    save_image(tempToSaveW/255,"temp_savedW.png")
            ##    save_image(tempToSaveD/255,"temp_savedD.png")
            #    self.savedATest == True
            #temp = np.reshape(temp,[-1,temp.shape[0],temp.shape[1]])
            #print("Temp shape = "+str(temp.shape))
            imgsPre.append(temp)
        #print("imgsPre.shape = "+str(torch.tensor(imgsPre).shape)) 
        imgsPre = torch.tensor(imgsPre)
        startSlice = int((imgsPre.shape[3]-self.minimumDims[2])/2)
        endSlice = startSlice + self.minimumDims[2]
        #print("startSlice = "+str(startSlice)+" and endSlice = "+str(endSlice))
        imgsPil = torch.zeros([self.modalityCount,self.minimumDims[1],self.minimumDims[0],self.minimumDims[2]])
        for i in range(0,self.modalityCount):
            for j in range(startSlice,endSlice):
                temp = Image.fromarray(imgsPre[i,:,:,j].numpy())
                temp = temp.resize((self.minimumDims[0],self.minimumDims[1]))
                #print("Temp shape = "+str(temp.size))
                imgsPil[i,:,:,j-startSlice] = transforms.ToTensor()(temp)
        
        imgs = imgsPil[:,:,:,:]*255
        
        labels = []
        if self.labelsOn == True:
            with open(self.labelData,'r') as readFile:
                reader = csv.reader(readFile, delimiter=',')
                rowCount = 0
                for row in reader:
                    if rowCount >0:
                        #print("row[0] = "+str(row[0])+" and id = "+str(self.folders[0][idx][23:30]))
                        if int(row[0]) == int(self.folders[0][idx][23:30]):
                            #print("Match Found")
                            for entry in row:
                                labels.append(int(entry))
                            break
                    rowCount += 1
            labels = torch.tensor(labels)
        #print("imgs.shape = "+str(imgs.shape))
        #Will return later to include demographics, medical data, etc. as vectors under labels
        
        #imgs = torch.zeros([self.modalityCount,self.minimumDims[2],self.minimumDims[1],self.minimumDims[0]])
        #if self.transforms:
        #    for i in range(0,self.modalityCount):
        #        for j in range(0,len(imgsPil[i])):
        #            imgs[i][j] = self.transforms(imgsPil[i][j]) 
        #else:
        #    print("No transforms provided to dataloader.")

        #imgsToSave = copy.deepcopy(imgs)
        #minImgs = torch.min(imgsToSave)
        #maxImgs = torch.max(imgsToSave)
        #imgsToSave = (imgsToSave-minImgs)/(maxImgs-minImgs)
        #imgsToSave = torch.reshape(imgsToSave[0,:,:,int(imgsToSave.shape[3]/2)],[1,imgsToSave.shape[1],imgsToSave.shape[2]])
        #save_image(torchvision.utils.make_grid(imgsToSave/255),"post_reshape_check.png")

        
        #print(str(len(imgs))+" modalities, and "+str(len(imgs[0]))+" slices in use.")
        #for i in range(0,len(imgs)):
        #    for j in range(0,len(imgs[0])):
        #        if torch.isnan(imgs[i][j]).any():
        #            print("NaN in image")
        #        for k in range(0,imgs[0][0].shape[0]):
        #            for l in range(0,imgs[0][0].shape[1]):
        #                if math.isnan(imgs[i][j][k][l]):
        #                    print("Nan in image")
        return imgs,labels



def filterLossFunc(outputs,inputs,biases,detLossScore):
	
	return biases[0].float()*nn.MSELoss(outputs,inputs) + biases[1].float()*detLossScore

def testImageReconstruction(net, testloader, device, minDims, fileName, slicePad):
     for batch in testloader:
        img, _ = batch
        #print("img shape = "+str(img.shape))
        #print("minDims = "+str(minDims))
        imgToSaveHeight = torch.reshape(img[:,0,int(minDims[1]/2),:,:],[img.size(0),1,minDims[0],minDims[2]])
        imgToSaveWidth = torch.reshape(img[:,0,:,int(minDims[0]/2),:],[img.size(0),1,minDims[1],minDims[2]])
        imgToSaveDepth = torch.reshape(img[:,0,:,:,int(minDims[2]/2)],[img.size(0),1,minDims[0],minDims[1]])
        #print("imgToSave shape = "+str(imgToSaveHeight.shape))
        save_image(torchvision.utils.make_grid(imgToSaveHeight),"checking_inputsH.png")
        save_image(torchvision.utils.make_grid(imgToSaveWidth),"checking_inputsW.png")
        save_image(torchvision.utils.make_grid(imgToSaveDepth),"checking_inputsD.png")
        img = img[:,:,:,:,int(minDims[2]/2)-slicePad:int(minDims[2]/2)+slicePad+1]
        img = img.to(device)
        #img = img.view(img.size(0), -1)
        outputs = net(img)
        outputs = outputs.cpu().data #outputs.view(outputs.size(0), outputs.size(1), minDims[1], minDims[0], minDims[2]).cpu().data
        outputsT1 = torch.reshape(outputs[:,0,:,:,slicePad],[outputs.size(0),1,minDims[1],minDims[0]])
        #outputsT2 = torch.reshape(outputs[:,1,int(minDims[0]/2),:,:],[outputs.size(0),1,minDims[1],minDims[2]])
        #print("outputsT1 type = "+str(outputsT1.dtype)+" and shape = "+str(outputsT1.shape))
        T1max = torch.max(outputsT1)
        T1min = torch.min(outputsT1)
        #T2max = torch.max(outputsT2)
        #T2min = torch.min(outputsT2)
        outputsT1 = ((outputsT1-T1min)/(T1max-T1min))
        #outputsT2 = ((outputsT2-T2min)/(T2max-T2min))

        #print("outputsT1 type is now "+str(outputsT1.dtype)+" and its max and min are "+str(torch.max(outputsT1))+" and "+str(torch.min(outputsT1)))
        #print("outputsT1 = "+str(outputsT1))
        #outputsT1 = Image.fromarray((outputsT1 * 255).astype(np.uint8))
        #outputsT2 = Image.fromarray((outputsT2 * 255).astype(np.uint8))
        #outputsT1 = ToPILImage()(outputsT1.astype(np.uint8))
        #outputsT2 = ToPILImage()(outputsT2.astype(np.uint8))
        #save_image(torch.rand(outputsT1.shape),"garbage_test.png")
        save_image(torchvision.utils.make_grid(outputsT1), fileName)
        #save_image(torchvision.utils.make_grid(outputsT2), 'T2_reconstruction.png')
        break


def computeOuterLinDim(batchSize,slicePad,convLayerCount,minImageDims,convChannels,convKernelSizes):
    height = minImageDims[0]
    width = minImageDims[1]
    depth = 2*slicePad+1
    channels = convChannels[0]

    for i in range(0,convLayerCount):
        depth = (depth - convKernelSizes[2])+1
        height = (height - convKernelSizes[0])+1
        width = (width - convKernelSizes[1])+1
        channels = convChannels[i+1]
    return depth*height*width*channels

#    for i in range(0,self.conv_layers):
#                 self.ae.append(nn.Conv3d(in_channels = kwargs["conv_channels"][i],out_channels = kwargs["conv_channels"][i+1],kernel_size = kwargs["conv_kernel_sizes"]))
#             
#             for i in range(0,self.conv_layers):
#                 self.ae.append(nn.ConvTranspose3d(in_channels = kwargs["conv_channels_backwards"][i],out_channels = kwargs["conv_channels_backwards"][i+1],kernel_size = kwargs["conv_kernel_sizes_backwards"]))



def fullTrain(testImageFileName = "check_learning_worked.png", bulkGradients = False, fullVolEval = False, dataSet = None, labels = 'ukb_Sex_BirthYear.csv', numEpochs = 20, learningRate = 1e-4, trainFrac = 0.9, runFrac = 1, batchSize = 24, slicePad = 4, codeDimension = 5000, convLayerCount = 4, minImageDims = [195,160,150], convChannels = [1,1,1,1,1],convKernelSizes = [3,3,3]):
        if bulkGradients == False: 
            learningRate = learningRate/(runFrac/0.1)
        #Defining the transforms we want performed on the data
        normalTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        
        preCodeDimension = computeOuterLinDim(batchSize,slicePad,convLayerCount,minImageDims[0:len(minImageDims)-1],convChannels,convKernelSizes)

        #If slicePad is less than the number of convolutional layers, you get problems. As such, we check for this and warn user.
        if slicePad*convKernelSizes[2] < (convLayerCount+1):
            print("Slice padding around central layer (for 2.5-dimensional image processing) is less than the number of convolutional layers in the encoder. This will cause a crash. Please decrease the number of layers, or increase slice padding value.")

        #Pick a device. If GPU available, use that. Otherwise, use CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Initialize a model of our autoEncoder class on the device
        ae = convAE(conv_layer_count=convLayerCount,conv_channels=convChannels, conv_kernel_sizes=convKernelSizes,conv_channels_backwards = convChannels[::-1],conv_kernel_sizes_backwards=convKernelSizes, linear_outer_feat = preCodeDimension, linear_inner_feat = codeDimension).to(device)
        #Define the optimization problem to be solved
        aeOpt = optim.Adam(ae.parameters(),lr=learningRate)
        #Define the objective function of the above optimization problem
        aeCriterion = nn.MSELoss()
        	
        #Loaders for the training and test datasets
        trainLoader = None
        testLoader = None

        if dataSet == None:
            totalData = brainImages(layersToInclude = 2*slicePad+1 ,minDims = minImageDims,labelFile = labels,transforms=normalTransform,pathsToData=["../../UKBiobankData/T1/"],modalities=["T1"])
            totalData.excludeLabels()
            
            trainData,testData = torch.utils.data.dataset.random_split(totalData,[int(len(totalData)*trainFrac),int(len(totalData))-int(len(totalData)*trainFrac)])
            trainFracs = [int(len(trainData)*runFrac),len(trainData)-int(len(trainData)*runFrac)]
            testFracs = [int(len(testData)*runFrac),len(testData)-int(len(testData)*runFrac)]
            trainData, _ = torch.utils.data.dataset.random_split(trainData,trainFracs)
            testData, _ = torch.utils.data.dataset.random_split(testData,testFracs)
        
            trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle=True,drop_last=True)
            testLoader = DataLoader(testData, batch_size = batchSize, shuffle = True,drop_last=True)
            torch.save(trainLoader,'trainLoader.pth')
            torch.save(testLoader,'testLoader.pth')
        else:
            trainLoader = torch.load('trainLoader.pth')
            testLoader = torch.load('testLoader.pth')
        print("There are "+str(len(trainLoader)*batchSize)+" training samples")	
        aeLossTracker = []

        #List of indices to cover, which will be shuffled for each batch
        shuffleOrder = []
        for i in range(slicePad, minImageDims[2]-slicePad):
            shuffleOrder.append(i)
         
        for epoch in range(numEpochs):
                print("Starting epoch "+str(epoch))
                aeLoss = 0
                for batch_features, labels in trainLoader:
                        #Reshape batch
                        #batch_features = batch_features.to(device)
                        random.shuffle(shuffleOrder)
                        #print("There are "+str(len(shuffleOrder))+" shuffled indices")
                        if bulkGradients == True:
                            aeOpt.zero_grad()
                            ae_train_loss_bulk = 0
                        for i in range(0,minImageDims[2]-2*slicePad-1):
                            #print("Middle slice is "+str(shuffleOrder[i]))
                            #print("batch_features shape = "+str(batch_features.shape))
                            temp_batch = batch_features[:,:,:,:,shuffleOrder[i]-slicePad:shuffleOrder[i]+slicePad+1]
                            #print("Dims of temp_batch = " +str(temp_batch.shape))
                            if torch.isnan(temp_batch).any():
                                print("Something wrong with temp_batch")
                            #print("Temp_batch shape = "+str(temp_batch.shape))
                            temp_batch = temp_batch.to(device)
                            if bulkGradients == False:
                                aeOpt.zero_grad()
                            aeOutputs = ae(temp_batch)
                            #print("AE output shape = "+str(aeOutputs.shape))
                            #aeOutputs = math.nan
                            #aeOutputs = torch.tensor(aeOutputs).to(device)
                            if torch.isnan(aeOutputs).any():
                                print("Error in network output")
                            if bulkGradients == False:
                                if fullVolEval == False:
                                    ae_train_loss = aeCriterion(aeOutputs[:,:,:,:,slicePad],temp_batch[:,:,:,:,slicePad])
                                    ae_train_loss.backward()
                                    aeOpt.step()
                                    aeLoss += ae_train_loss.item()
                                    if torch.isnan(ae_train_loss).any():
                                        print("Something wrong with loss signal")
                                        exit()
                                else:
                                    ae_train_loss = aeCriterion(aeOutputs[:,:,:,:,:],temp_batch[:,:,:,:,:])/(2*slicePad + 1)
                                    ae_train_loss.backward()
                                    aeOpt.step()
                                    aeLoss += ae_train_loss.item()
                                    if torch.isnan(ae_train_loss).any():
                                        print("Something wrong with loss signal")
                                        exit()
                            else:
                                temp_loss= aeCriterion(aeOutputs[:,:,:,:,slicePad],temp_batch[:,:,:,:,slicePad])/minImageDims[2]
                                aeLoss += temp_loss.item()
                                ae_train_loss_bulk= temp_loss
                                #print("ae_train_loss = "+str(ae_train_loss.item()))
                                if torch.isnan(ae_train_loss_bulk).any():
                                    print("Something is wrong with loss signal")
                                    exit()
                            #print("ae_train_loss.item() = "+str(ae_train_loss.item()))
                        if bulkGradients == True:
                            ae_train_loss_bulk.backward()
                            aeOpt.step()
                            #aeLoss = ae_train_loss.item()
                #Normalize loss for epoch
                aeLoss = aeLoss/(len(trainLoader)*batchSize*runFrac)
                aeLossTracker.append(aeLoss)
                #Print epoch num and corresponding loss
                print("Epoch: {}/{}, autoencoder loss = {:6f}".format(epoch+1,numEpochs,aeLoss))
                testImageReconstruction(ae,testLoader,device,minImageDims,"training_test_epoch_"+str(epoch)+".png",slicePad) 
	

        plt.figure()
        plt.plot(aeLossTracker)
        plt.title('Training Loss')
        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.savefig('ae_deep_fashionMNIST_loss.png')
       
        torch.save(ae,"brain_ae_model_"+str(preCodeDimension)+"_codeDim_"+str(runFrac)+"_runFrac_"+str(fullVolEval)+"_fullVolEval.pt")

        with open('saved_loss_trajectory_'+str(preCodeDimension)+'_codeDim_'+str(runFrac)+'_runFrac.csv',mode='w') as trajectoryFile:
            writer = csv.writer(trajectoryFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(aeLossTracker)


        testImageReconstruction(ae,testLoader,device,minImageDims,testImageFileName,slicePad)

	

# %%
