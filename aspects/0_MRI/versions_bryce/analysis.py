import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import brainAE as brainCore
import copy
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import statsmodels.stats.multitest as multitest


#matplotlib inline
import matplotlib
import analysisUtil as aUtil

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)






def analyze(modelToLoad = None, codeDim = None, labels = None, plotTitles = None, UKBDataFile = None, labelsToChunk = None, model = None, uniformWeighting = True, neighborFrac = 0.75, chunkiles = 5, UMAPneighborsFrac = 0.2, neighborCount=15, maxEmbedDim = 10, weightingWidthFactor = 1, newDataLoader = True, runFrac=0.1,trainFrac=0.01,minImageDims=[195,160,150],batchSize = 24, slicePad = 2):
    

    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if newDataLoader == False:
        trainLoader = torch.load('trainLoader.pth')
        testLoader = torch.load('testLoader.pth')
    else:
        totalData = brainCore.brainImages(layersToInclude = 2*slicePad+1 ,minDims = minImageDims,labelFile = UKBDataFile,transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]),pathsToData=["../../UKBiobankData/T1/"],modalities=["T1"])
        trainData,testData = torch.utils.data.dataset.random_split(totalData,[int(len(totalData)*trainFrac),int(len(totalData))-int(len(totalData)*trainFrac)])
        testData = totalData
        trainFracs = [int(len(trainData)*runFrac),len(trainData)-int(len(trainData)*runFrac)]
        testFracs = [int(len(testData)*runFrac),len(testData)-int(len(testData)*runFrac)]
        trainData, _ = torch.utils.data.dataset.random_split(trainData,trainFracs)
        testData, _ = torch.utils.data.dataset.random_split(testData,testFracs)
        
        trainLoader = DataLoader(trainData, batch_size = batchSize, shuffle=True,drop_last=True)
        testLoader = DataLoader(testData, batch_size = batchSize, shuffle = True,drop_last=True)
    ae = None
    if modelToLoad==None:
        ae = torch.load(model).to(device)
    else:
        ae = modelToLoad
    sliceCount = 0
    for batch_features,_ in testLoader:
        sliceCount = batch_features.shape[4]
        break
    fullData = torch.zeros([batchSize*len(testLoader),codeDim])
    fullLabels = torch.zeros([batchSize*len(testLoader),labels+1])
    k=0
    p=0
    for batch_features,labels in testLoader:
        batchData = torch.zeros([batchSize,sliceCount-(2*slicePad)-1,codeDim])
        batchLabels = torch.zeros([batchSize,sliceCount-(2*slicePad)-1,len(plotTitles)+1])
        for i in range(0,batch_features.shape[4]-(2*slicePad)-1):
            temp_batch = batch_features[:,:,:,:,i:i+(2*slicePad)+1].to(device)
            code = ae.getCode(temp_batch)[:,:,:,:,:].detach()
            #print("Code shape = "+str(code.shape))
            #print("Labels shape = "+str(labels.shape))
            for j in range(0,code.shape[0]):
                if int(labels[j,2]) == 0:
                    continue
                else:
                    batchData[j,i,:] = torch.flatten(copy.deepcopy(code[j,:,:,:,:]))
                    batchLabels[j,i,:] = torch.tensor(labels[j])

            del temp_batch
            del code
        if uniformWeighting == True:
            for i in range(0,batchSize):
                #Taking mean of the codes across the brain slices
                fullData[p+i,:] = torch.mean(batchData[i,:,:],0)
                fullLabels[p+i,:] = torch.mean(batchLabels[i,:,:],0)
                #print("Age labels for this patient = "+str(batchLabels[i,:,2]))
                #print("fullData memory = "+str(fullData.nelement()*fullData.element_size()))
        else:
            weightSum = 0
            weightWidthFactorTemp = weightingWidthFactor*(batch_features.shape[4]-(2*slicePad)-1)
            for i in range(0,batchSize):
                for j in range(0,batch_features.shape[4]-(2*slicePad)-1):
                    weight = np.exp(-(int(j-(batch_features.shape[4]-(2*slicePad)-1)/2)**2)/weightWidthFactorTemp)
                    weightSum += weight
                    fullData[p+i,:] += torch.mean(weight*batchData[i,:,:],0)
                fullData[p+i,:] = fullData[p+i,:]/weightSum
                fullLabels[p+i,:] = torch.mean(batchLabels[i,:,:],0)
        del batchData
        del batchLabels
        p+=batchSize
        print("Datapoints so far = "+str(p))
    





    #UMAP and other embedding analyses
    n_neighbors = neighborCount
    X = []
    y = []
    yChunk = []
    indices = []
    KNNLabels = []
    #Testing out parameters
    UMAPneighbors = int(p*UMAPneighborsFrac)
    for embeddingDim in range(maxEmbedDim,1,-1):
        print("Analyzing with embedding dimension "+str(embeddingDim))
        #print("FullData.shape = "+str(fullData.shape))
        #Computing the various embeddings to try
        reducer = umap.UMAP(n_components = embeddingDim,n_neighbors=UMAPneighbors)
        embeddingUMAP = reducer.fit_transform(fullData[:,:])
        embeddingTSNE = None
        if embeddingDim <4:
            embeddingTSNE = TSNE(n_components=embeddingDim).fit_transform(fullData[:,:])
        embeddingPCA = PCA(n_components=embeddingDim).fit_transform(fullData[:,:])
        #print("FullLabels.shape = "+str(fullLabels.shape))
        #print("ones.shape = "+str(torch.ones(fullLabels.shape[0]).shape))
        fullLabels[:,2] = torch.mul(torch.ones(fullLabels.shape[0]),2021)-fullLabels[:,2]

        #print("Ages range from "+str(torch.min(fullLabels[:,2]))+" to "+str(torch.max(fullLabels[:,2])))
        #print("Genders = "+str(fullLabels[:,1]))
        #print("Ages = "+str(fullLabels[:,2]))
        chunkLabels = copy.deepcopy(fullLabels)
        if chunkiles > 0:
            for i in labelsToChunk:
                chunkLabels[:,i] = torch.tensor(pd.qcut(pd.Series(fullLabels[:,i]).rank(method='first'),chunkiles,labels=False))
            #print("Age buckets = "+str(fullLabels[:,2]))
            
        #Here's where we do KNN on the embeddings


        neighborEnd = int(neighborFrac*embeddingUMAP.shape[0])
        X = embeddingUMAP[0:neighborEnd,:]
        XPCA = embeddingPCA[0:neighborEnd,:]
        XTSNE = None
        if embeddingDim<4:
            XTSNE = embeddingTSNE[0:neighborEnd,:]
        y=[]#torch.zeros([fullLabels.shape[1]-1,neighborEnd])
        for i in range(1,fullLabels.shape[1]):
            y.append(fullLabels[0:neighborEnd,i])
        
        yChunk = copy.deepcopy(y)
        for i in labelsToChunk:
            yChunk[i-1] = chunkLabels[:neighborEnd,i]
        #KNN inference accuracy calculation
        KNNUMAPaccuracies = []
        KNNPCAaccuracies = []
        KNNTSNEaccuracies = []
        indices = []
        KNNLabels =[]
        for k in range(0,neighborCount):
            for i in range(1,fullLabels.shape[1]):
                modelUMAP = neighbors.KNeighborsClassifier(k+1)
                modelPCA = neighbors.KNeighborsClassifier(k+1)
                
                modelUMAP.fit(X,yChunk[i-1])
                modelPCA.fit(XPCA,yChunk[i-1])
                
                if embeddingDim<4:
                    modelTSNE = neighbors.KNeighborsClassifier(k+1)
                    modelTSNE.fit(XTSNE,yChunk[i-1])
                    predTSNE = modelTSNE.predict(embeddingTSNE[neighborEnd:,:])
                    KNNTSNEaccuracies.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predTSNE))
                predUMAP = modelUMAP.predict(embeddingUMAP[neighborEnd:,:])
                predPCA = modelPCA.predict(embeddingPCA[neighborEnd:,:])
                

                KNNUMAPaccuracies.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predUMAP))
                KNNPCAaccuracies.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predPCA))
                
                indices.append(k*(fullLabels.shape[1]-1)+i)
                if i == 1:
                    KNNLabels.append(str(k+1))#str(k)+"-"+plotTitles[i-1])
                else:
                    KNNLabels.append(' ')
        UMAPforestR2 = []
        UMAPforestP = []
        UMAPlinearR2 = []
        UMAPlinearP = []
        TSNEforestR2 = []
        TSNEforestP = []
        TSNElinearR2 = []
        TSNElinearR = []

        regIndices = []
        XChunks = []
        XMeans = []
        XGlobalMeans = []
        for i in range(1,fullLabels.shape[1]):
            UMAPRegForest = RandomForestRegressor(max_depth=2, random_state=0)
            UMAPRegLinear = LinearRegression()

            UMAPRegForest.fit(X,y[i-1])
            UMAPRegLinear.fit(X,y[i-1])
            UMAPforestR2.append(copy.deepcopy(UMAPRegForest.score(X,y[i-1])))
            UMAPlinearR2.append(copy.deepcopy(UMAPRegLinear.score(X,y[i-1])))

            TSNERegForest = RandomForestRegressor(max_depth=2,random_state=0)
            TSNERegLinear = LinearRegression()

            TSNERegForest.fit(X,y[i-1])
            TSNERegLinear.fit(X,y[i-1])
            TSNEforestR2.append(copy.deepcopy(TSNERegForest.score(X,y[i-1])))
            TSNElinearR2.append(copy.deepcopy(TSNERegLinear.score(X,y[i-1])))
            
            regIndices.append(i)
        


        UMAPKNNTitle = "UMAP KNN accuracies dim="+str(embeddingDim)+" neighbors ratio="+str(UMAPneighborsFrac)+" "
        PCAKNNTitle = "PCA KNN accuracies dim="+str(embeddingDim)+" "
        TSNEKNNTitle = "t-SNE KNN accuracies dim="+str(embeddingDim)+" "
        UMAPForRegTitle = "UMAP Forest Regression r^2, dim="+str(embeddingDim)+" "
        TSNEForRegTitle = "t-SNE Forest Regression r^2, dim="+str(embeddingDim)+" "
        UMAPLinRegTitle = "UMAP Linear Regression r^2, dim="+str(embeddingDim)+" "
        TSNELinRegTitle = "t-SNE Linear Regression r^2, dim="+str(embeddingDim)+" "


        for i in range(1,fullLabels.shape[1]):
            UMAPKNNTitle = UMAPKNNTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            PCAKNNTitle = PCAKNNTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            TSNEKNNTitle = TSNEKNNTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            UMAPForRegTitle = UMAPForRegTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            TSNEForRegTitle = TSNEForRegTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            UMAPLinRegTitle = UMAPLinRegTitle+ str(i)+"th is "+plotTitles[i-1]+" "
            TSNELinRegTitle = TSNELinRegTitle+ str(i)+"th is "+plotTitles[i-1]+" "

        
        #Plotting KNN accuracies by label
        plt.bar(indices,KNNUMAPaccuracies,align='center')
        plt.xticks(indices,KNNLabels,wrap=True)
        plt.title(UMAPKNNTitle,wrap=True)
        plt.savefig('UMAP_KNNaccuracies_embDim_'+str(embeddingDim)+'_neighborFrac_'+str(UMAPneighborsFrac)+'.png')
        plt.clf()

        plt.bar(indices,KNNPCAaccuracies,align='center')
        plt.xticks(indices,KNNLabels,wrap=True)
        plt.title(PCAKNNTitle,wrap=True)
        plt.savefig('PCA_KNNaccuracies_embDim_'+str(embeddingDim)+'.png')
        plt.clf()
        if embeddingDim<4:
            plt.bar(indices,KNNTSNEaccuracies,align='center')
            plt.xticks(indices,KNNLabels,wrap=True)
            plt.title(TSNEKNNTitle,wrap=True)
            plt.savefig('t-SNE_KNNaccuracies_embDim_'+str(embeddingDim)+'.png')
            plt.clf()

        plt.bar(regIndices,UMAPforestR2,align='center')
        plt.xticks(regIndices,plotTitles,wrap=True)
        plt.title(UMAPForRegTitle,wrap=True)
        plt.savefig('UMAP_Forest_Regression_R2_embDim_'+str(embeddingDim)+'.png')
        plt.clf()

        plt.bar(regIndices,UMAPlinearR2,align='center')
        plt.xticks(regIndices,plotTitles,wrap=True)
        plt.title(UMAPLinRegTitle,wrap=True)
        plt.savefig('UMAP_Linear_Regression_R2_embDim_'+str(embeddingDim)+'.png')
        plt.clf()

        plt.bar(regIndices,TSNEforestR2,align='center')
        plt.xticks(regIndices,plotTitles,wrap=True)
        plt.title(TSNEForRegTitle,wrap=True)
        plt.savefig('t-SNE_Forest_Regression_R2_embDim_'+str(embeddingDim)+'.png')
        plt.clf()

        plt.bar(regIndices,TSNElinearR2,align='center')
        plt.xticks(regIndices,plotTitles,wrap=True)
        plt.title(TSNELinRegTitle,wrap=True)
        plt.savefig('t-SNE_Linear_Regression_R2_embDim_'+str(embeddingDim)+'.png')
        plt.clf()


    ##Computing p-values
    #Segnent data by label
    for i in range(1,fullLabels.shape[1]):
        print("i =" +str(i))
        XChunks.append([])
        if i in labelsToChunk:
            for j in range(0,chunkiles):
                XChunks[i-1].append([])
                for k in range(0,fullData.shape[0]):
                    if chunkLabels[k,i-1] == j:
                        #print("Adding from chunkiles")
                        XChunks[i-1][j].append(fullData[k,:])
        else:
            maxIndex,_ = torch.max(fullLabels[:,i],0)
            print(str(int(maxIndex)))
            for j in range(0,int(maxIndex)):
                XChunks[i-1].append([])
                for k in range(0,fullData.shape[0]):
                    #print("Adding from non-chunked labels")
                    XChunks[i-1][j].append(fullData[k,:])
    #Find means for each value
    #For each label
    for i in range(1,fullLabels.shape[1]):
        XMeans.append([])
        if i in labelsToChunk:
            #For each possible value for that label
            for j in range(0,chunkiles):
                XMeans[i-1].append([])
                #For each code variable
                for m in range(0,fullData.shape[1]):
                    XMeans[i-1][j].append(0)
                    count = 0
                    #For each patient
                    for k in range(0,len(XChunks[i-1][j])):
                        XMeans[i-1][j][m]+=XChunks[i-1][j][k][m]
                        count +=1
                    if count>0:
                        XMeans[i-1][j][m] = XMeans[i-1][j][m]/count

        else:
            #For each possible value for that label
            maxIndex,_ = torch.max(fullLabels[:,i],0)
            for j in range(0,int(maxIndex)):
                XMeans[i-1].append([])
                #For each code variable
                for m in range(0,fullData.shape[1]):
                    XMeans[i-1][j].append(0)
                    count=0
                    #For each patient
                    for k in range(0,len(XChunks[i-1][j])):
                        XMeans[i-1][j][m]+=XChunks[i-1][j][k][m]
                        count+=1
                    if count>0:
                        XMeans[i-1][j][m] = XMeans[i-1][j][m]/count
        
    #Compute variances for each label and code dimension
    #print("XMeans = "+str(XMeans))
    XVars=[]
    for i in range(1,fullLabels.shape[1]):
        XVars.append([])
        if i in labelsToChunk:
            #For each possible value for that label
            for j in range(0,chunkiles):
                XVars[i-1].append([])
                #For each code variable
                for m in range(0,fullData.shape[1]):
                    XVars[i-1][j].append(0)
                    count = 0
                    #For each patient
                    for k in range(0,len(XChunks[i-1][j])):
                        XVars[i-1][j][m]+=(XChunks[i-1][j][k][m]-XMeans[i-1][j][m])**2
                        count += 1
                    if count>0:
                        XVars[i-1][j][m] = XVars[i-1][j][m]/count
        else:
            maxIndex,_ = torch.max(fullLabels[:,i],0)
            for j in range(0,int(maxIndex)):
                XVars[i-1].append([])
                for m in range(0,fullData.shape[1]):
                    XVars[i-1][j].append(0)
                    count = 0
                    for k in range(0,len(XChunks[i-1][j])):
                        XVars[i-1][j][m]+=(XChunks[i-1][j][k][m]-XMeans[i-1][j][m])**2
                        count+=1
                    if count>0:
                        XVars[i-1][j][m] = XVars[i-1][j][m]/count

    #print("XVars = "+str(XVars))
    
    #Compute means for each code dimension
    for m in range(0,fullData.shape[1]):
        XGlobalMeans.append(0)
        count = 0
        #For each patient
        for k in range(0,fullData.shape[0]):
            XGlobalMeans[m]+= fullData[k,m]
            count += 1
        if count>0:
            XGlobalMeans[m] = XGlobalMeans[m]/count
    #print("XGlobalMeans = "+str(XGlobalMeans))
    XGlobalVars=[]
    #Compute variances for each code dimension
    for m in range(0,fullData.shape[1]):
        XGlobalVars.append(0)
        count = 0
        #For each patient
        for k in range(0,fullData.shape[0]):
            XGlobalVars[m]+=(fullData[k,m]-XGlobalMeans[m])**2
            count+=1
        if count>0:
            XGlobalVars[m] = XGlobalVars[m]/count
    #print("XGlobalVars = "+str(XGlobalVars))
    #For each label value and dimension of code, compute p-value
    pValues = []
    pValsFlat = []
    pIndices = []
    for i in range(1,fullLabels.shape[1]):
        pValues.append([])
        for j in range(0,len(XMeans[i-1])):
            pValues[i-1].append([])
            for m in range(0,fullData.shape[1]):
                s = np.sqrt((XGlobalVars[m]+XVars[i-1][j][m])/2)
                t = (XMeans[i-1][j][m] - XGlobalMeans[m])/(s*np.sqrt(2/fullData.shape[0]))
                df = 2*(fullData.shape[0]-1)
                p = 2 - 2*stats.t.cdf(t,df=df)
                pValues[i-1][j].append(p)
                pValsFlat.append(p)
                pIndices.append([i,j,m])
    #print("pValues = "+str(pValues))
    #print("pValsFlat = "+str(pValsFlat))
    #print("pIndices = "+str(pIndices))

    pValsFlatNumpy = np.array(pValsFlat)
    sort_index = np.argsort(pValsFlatNumpy)
    pValsFlatSort = np.sort(pValsFlatNumpy)
    print("pValsFlatSort = "+str(pValsFlatSort))
    rejections,pVals_corrected,_,_ = multitest.multipletests(pValsFlatSort,method='fdr_bh')
    rejections = rejections.tolist()
    pVals_corrected = pVals_corrected.tolist()
    #print("rejections = "+str(rejections))

    pValsNullReject = []
    pValsNullRejectIndices=[]
    for i in range(0,len(pVals_corrected)):
        if rejections[i]==True:
            print("Adding a reject")
            pValsNullReject.append(pVals_corrected[i])
            pValsNullRejectIndices.append(pIndices[sort_index[i]])
    print("pValsNullRejectIndices = "+str(pValsNullRejectIndices))
    numNullRejectsByLabel=[]
    rejectsIndices = []
    for i in range(1,fullLabels.shape[1]):
        numNullRejectsByLabel.append(0)
        rejectsIndices.append(i-1)
        for indexes in pValsNullRejectIndices:
            if indexes[0] == i:
                numNullRejectsByLabel[i-1]+=1
    print("numNullRejectsByLabel = "+str(numNullRejectsByLabel))

    rejectsTitleByLabel = "Number of significant dimensions per feature of interest, "
    for i in range(0,len(plotTitles)):
        rejectsTitleByLabel = rejectsTitleByLabel + str(i)+'th label is '+plotTitles[i]+" "
    
    plt.bar(rejectsIndices,numNullRejectsByLabel,align='center')
    plt.xticks(rejectsIndices,plotTitles,wrap=True)
    plt.title(rejectsTitleByLabel,wrap=True)
    plt.savefig('num_significant_code_dims_per_feature.png')
    plt.clf()


    numNullRejectsByLabelVal=[]
    rejectsIndices = []
    for i in range(1,fullLabels.shape[1]):
        numNullRejectsByLabelVal.append([])
        rejectsIndices.append([])
        for j in range(0,len(XMeans[i-1])):
            print("j = "+str(j))
            rejectsIndices[i-1].append(j)
            numNullRejectsByLabelVal[i-1].append(0)
            for indices in pValsNullRejectIndices:
                if indices[0] == i and indices[1]==j:
                    numNullRejectsByLabelVal[i-1][j]+=1
    print("numNullRejects = "+str(numNullRejectsByLabelVal))
    for i in range(0,len(plotTitles)):
        tempTitle = "Number of significant dimensions per label for "+plotTitles[i]
        plt.bar(rejectsIndices[i],numNullRejectsByLabelVal[i],align='center')
        plt.title(tempTitle,wrap=True)
        plt.savefig('num_significant_code_dims_for_'+plotTitles[i]+'.png')
        plt.clf()

    print("Running KNN on full, unmapped code")
    XFull = fullData[0:neighborEnd,:]
    fullRand = torch.rand(fullData.shape)
    XRand = fullRand[0:neighborEnd,:]
    KNNFullDataAccuracies = []
    KNNRandDataAccuracies = []
    KNNFullDataAccTensor = torch.zeros([fullLabels.shape[1]-1,neighborCount])
    indices = []
    KNNLabels = []
    for k in range(0,neighborCount):
        print("Full, unmapped code with clustering k="+str(k+1))
        for i in range(1,fullLabels.shape[1]):
            modelFull = neighbors.KNeighborsClassifier(k+1)
            modelRand = neighbors.KNeighborsClassifier(k+1)
            modelFull.fit(XFull,yChunk[i-1])
            modelRand.fit(XRand,yChunk[i-1])
            predFull = modelFull.predict(fullData[neighborEnd:,:])
            predRand = modelRand.predict(fullRand[neighborEnd:,:])
            KNNFullDataAccTensor[i-1,k] = aUtil.accuracy(chunkLabels[neighborEnd:,i],predFull)
            KNNFullDataAccuracies.append(KNNFullDataAccTensor[i-1,k])
            KNNRandDataAccuracies.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predRand))
            indices.append(k*(fullLabels.shape[1]-1)+i)
            if i==1:
                KNNLabels.append(str(k+1))
            else:
                KNNLabels.append(' ')
    
    #SVM Classification test
    SVMFullDataAccuracies = []
    SVMLabels = []
    SVMIndices = []
    for i in range(1,fullLabels.shape[1]):
        modelSVM = svm.SVC()
        fitSVM = modelSVM.fit(XFull,yChunk[i-1])
        predSVM = modelSVM.predict(fullData[neighborEnd:,:])
        SVMFullDataAccuracies.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predSVM))
        SVMIndices.append(i)
        SVMLabels.append(plotTitles[i-1])
    
    SVMRBFFullDataAcc = []
    
    for expoGam in range(-1,3):
        for expoC in range(-1,3):
            SVMRBFFullDataAcc=[]
            for i in range(1,fullLabels.shape[1]):
                modelSVM = svm.SVC(gamma=10**expoGam,C=10**expoC)
                fitSVM = modelSVM.fit(XFull,yChunk[i-1])
                predSVM = modelSVM.predict(fullData[neighborEnd:,:])
                SVMRBFFullDataAcc.append(aUtil.accuracy(chunkLabels[neighborEnd:,i],predSVM))
            SVMRBFTitle = "Full Code SVM accuracies, RBF kernel, dim="+str(codeDim)+",gamma="+str(10**expoGam)+", c="+str(10**expoC)
            plt.bar(SVMIndices,SVMRBFFullDataAcc,align='center')
            plt.xticks(SVMIndices,SVMLabels,wrap=True)
            plt.title(SVMRBFTitle,wrap=True)
            plt.savefig('Full_SVM_RBFKernel_gam='+str(10**expoGam)+'_c='+str(10**expoC)+'.png')
            plt.clf()
        

    
    FullKNNTitle = "Full code KNN accuracies dim="+str(codeDim)+" "
    plt.bar(indices,KNNFullDataAccuracies,align='center')
    plt.xticks(indices,KNNLabels,wrap=True)
    plt.title(FullKNNTitle,wrap=True)
    plt.savefig('Full_KNNaccuracies.png')
    plt.clf()

    RandKNNTitle = "Random code KNN accuracies dim="+str(codeDim)+" "
    plt.bar(indices,KNNRandDataAccuracies,align='center')
    plt.xticks(indices,KNNLabels,wrap=True)
    plt.title(RandKNNTitle,wrap=True)
    plt.savefig('Rand_KNNaccuraciesTest.png')
    plt.clf()

    SVMTitle = "Full code SVM accuracies dim="+str(codeDim)+" "
    plt.bar(SVMIndices,SVMFullDataAccuracies,align='center')
    plt.xticks(SVMIndices,SVMLabels,wrap=True)
    plt.title(SVMTitle,wrap=True)
    plt.savefig('Full_SVMaccuracies.png')
    plt.clf()




    #Miscellania for the KNN visualizations
    h = .02  # step size in the mesh

    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)


    #Producing remaining plots
    for i in range(0,len(plotTitles)):
        
        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, yChunk[i])
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=light_jet)
            # Plot also the training points
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y[i],
                    cmap=matplotlib.cm.jet, alpha=1.0, edgecolor="black")
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title('UMAP '+plotTitles[i]+" KNN classification (k = %i, weights = '%s')"
                % (n_neighbors, weights))
            plt.savefig('UMAP '+plotTitles[i]+" KNN classification (k = %i, weights = '%s')"
                % (n_neighbors, weights))
        plt.clf()

        

        #Making scatter plots for each embedding
        plt.scatter(
            embeddingUMAP[:, 0],
            embeddingUMAP[:, 1],
            c=fullLabels[:,i+1].int(), cmap='Spectral')#, s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP '+plotTitles[i], fontsize=24)
        plt.savefig('UMAP '+plotTitles[i]+'.png')
        plt.clf()
        plt.scatter(
            embeddingTSNE[:,0],
            embeddingTSNE[:,1],
            c=fullLabels[:,i+1].int(), cmap='Spectral')
        plt.gca().set_aspect('equal','datalim')
        plt.title('TSNE '+plotTitles[i],fontsize=24)
        plt.savefig('TSNE '+plotTitles[i]+'.png')
        plt.clf()
        plt.scatter(
            embeddingPCA[:,0],
            embeddingPCA[:,1],
            c=fullLabels[:,i+1].int(), cmap='Spectral')
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('PCA '+plotTitles[i], fontsize=24)
        plt.savefig('PCA '+plotTitles[i]+'.png')
        plt.clf()


    return KNNFullDataAccTensor


#Labels to chunk should be indexed from 1, since the first "label" is actually the patient ID
analyze(codeDim = 28424, labels=5, slicePad = 4, uniformWeighting = True, labelsToChunk = [2,3,4,5], plotTitles = ['sex','age','BrainVolume','WhiteMatterVolume','GreyMatterVolume'],UKBDataFile = 'ukb_Sex_BirthYear_BrainVolume_WhiteMatterVolume_GreyMatterVolume.csv',model = 'brain_ae_model_28424_codeDim_1_runFrac_False_fullVolEval.pt')

