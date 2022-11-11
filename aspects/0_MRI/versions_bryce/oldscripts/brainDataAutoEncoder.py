import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image



class autoEncoder(nn.Module):
	def __init__(self,**kwargs):
		super().__init__()
		#Creating the neural network structure
		
		#Encoder portion of the autoencoder
		self.encoderHiddenLayer = nn.Linear(in_features = kwargs["input_shape"], out_features = kwargs["middle_dimension_outer"])
		self.encoderHiddenLayer2 = nn.Linear(in_features = kwargs["middle_dimension_outer"], out_features = kwargs["middle_dimension_inner"])
		self.encoderOutputLayer = nn.Linear(in_features = kwargs["middle_dimension_inner"], out_features = kwargs["encode_dimension"])
		
		#Decoder portion
		self.decoderHiddenLayer = nn.Linear(in_features = kwargs["encode_dimension"], out_features = kwargs["middle_dimension_inner"])
		self.decoderHiddenLayer2 = nn.Linear(in_features = kwargs["middle_dimension_inner"], out_features = kwargs["middle_dimension_outer"])
		self.decoderOutputLayer = nn.Linear(in_features = kwargs["middle_dimension_outer"], out_features = kwargs["input_shape"])
		
	def forward(self, features):
		#Defining the forward processing of the full autoencoder neural network
		activation = self.encoderHiddenLayer(features)
		activation = torch.relu(activation)
		activation = self.encoderHiddenLayer2(activation)
		activation = torch.relu(activation)
		code = self.encoderOutputLayer(activation)
		code = torch.relu(code)
		activation = self.decoderHiddenLayer(code)
		activation = torch.relu(activation)
		activation = self.decoderHiddenLayer2(activation)
		activation = torch.relu(activation)
		activation = self.decoderOutputLayer(activation)
		reconstruction = torch.relu(activation)
		#Returning the encoded data and the reconstructed data
		return reconstruction

def makeDir():
	image_dir = 'FashionMNISTImages'
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

def saveDecodedImage(img,epoch):
	img = img.view(img.size(0),1,28,28)
	save_img(img,'./FashionMNISTImages/linearAEImage{}.png'.format(epoch))


def encodeDecode(numEpochs = 50, learningRate = 1e-3, batchSize = 128, inputShape=784, middleDimension=256, encodeDimension=128):
	#Pick a device. If GPU available, use that. Otherwise, use CPU.
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#Initialize a model of our autoEncoder class on the device
	model = autoEncoder(input_shape = inputShape, middle_dimension_inner = middleDimension, middle_dimension_outer = middleDimension, encode_dimension = encodeDimension).to(device)

	#Define the optimization problem to be solved
	optimizer = optim.Adam(model.parameters(),lr=1e-3)
	
	#Define the objective function of the above optimization problem
	criterion = nn.MSELoss()



	#Testing this code with some toy data
	#Defining the transforms we want performed on the data
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
	#Defining the training set
	trainSet = datasets.FashionMNIST(root='./data',train = True,download = True, transform = transform)
	#Defining the testing set
	testSet = datasets.FashionMNIST(root='./data',train=False,download = True, transform = transform)
	
	trainLoader = DataLoader(trainSet, batch_size = batchSize, shuffle=True)
	testLoader = DataLoader(testSet, batch_size = batchSize, shuffle = True)
	
	lossTracker = []
	for epoch in range(numEpochs):
		loss = 0
		for batch_features, _ in trainLoader:
			#Reshape batch
			batch_features = batch_features.view(-1,inputShape).to(device)

			#Reset gradients to zero
			optimizer.zero_grad()
			
			#Compute reconstructions
			outputs = model(batch_features)

			#Compute training loss
			train_loss = criterion(outputs,batch_features)

			#Compute gradients
			train_loss.backward()

			#Update parameters
			optimizer.step()
			
			#Add batch training loss to epoch loss 
			loss += train_loss.item()
		
		#Normalize loss for epoch
		loss = loss/len(trainLoader)
		lossTracker.append(loss)
		#Print epoch num and corresponding loss
		print("Epoch: {}/{}, loss = {:6f}".format(epoch+1,numEpochs,loss))
		
	plt.figure()
	plt.plot(lossTracker)
	plt.title('Training Loss')
	plt.xlabel('Training Epochs')
	plt.ylabel('Loss')
	plt.savefig('deep_ae_fashionMNIST_loss.png')
		
		
encodeDecode()		
