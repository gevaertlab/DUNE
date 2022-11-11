import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
import torch
import numpy as np
import sklearn.manifold as manifold 
from brainAE import brainImages as loadData
import umap

def analyze(plotTitles = None, plotLabels=None, labelFile = None, minDims = None, batchSize = None, neighborFrac = None):

    dataSet = loadData(minDims,labelFile,pathsToData=["../../UKBiobankData/T1/"],modalities=["T1"])

    trainLoader = DataLoader(dataSet,batch_size = batchSize, shuffle = True, drop_last=True)
    
    batchData = torch.zeros([batchSize,np.prod(minDims)])
    batchLabels = torch.zeros([batchSize,len(plotTitles)+1])
    for batch_features,labels in trainLoader:
        for j in range(0,batchSize):
                if int(labels[j,2]) == 0:
                    continue
                else:
                    batchData[j,:] = torch.flatten(batch_features[j]))
                    batchLabels[j,:] = torch.tensor(labels[j])

        break

    neighborEnd = np.floor(batchSize*neighborFrac)
    for j in range(2,maxEmbDim):
        for i in range(0,len(plotTitles)
            for k in range(1,maxEmbNeighbors):
                y = labels[:neighborEnd,i+1]

                ISO = manifold.Isomap(n_components = j,n_neighbors=k)
                embeddingISO = ISO.fit_transform(batchData)
                X = embeddingISO[:neighborEnd,:]

                if j==2:
                    for weights in ['uniform', 'distance']:
                        # we create an instance of Neighbours Classifier and fit the data.
                        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                        clf.fit(X, y)
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
                        plt.title('ISOMAP '+plotTitles[i]+" KNN classification (k = %i, weights = '%s')"
                            % (n_neighbors, weights))
                        plt.savefig('ISOMAP '+plotTitles[i]+" KNN classification (k = %i, weights = '%s').png"
                            % (n_neighbors, weights))
                    plt.clf()
                KNNISOaccuracies = []
                indices = []
                for n in range(1,maxKNN_neighbors):
                    modelISO = neighbors.KNeighborsClassifier(n+1)
                    modelISO.fit(X,y)
                    predISO = modelISO.predict(embeddingISO[neighborEnd:,:])
                    KNNISOaccuracies.append(aUtil.accuracy(fullLabels[neighborEnd:,i],predISO))
                    indices.append(n)


                plt.bar(indices,KNNISOaccuracies,align='center')
                #plt.xticks(indices,KNNLabels)
                plt.title('ISOMAP KNN accuracies w.r.t'+plotLabels[i]+', emb dim='+str(j))
                plt.savefig('ISOMAP_KNNaccuracies_fullImage_'+plotTitles[i]+'_embDim_'+str(j)+'_neighborFrac_'+str(UMAPneighborsFrac)+'.png')
                plt.clf()





                UMAP = umap.UMAP(n_components = j,n_neighbors=k) 
                embeddingUMAP = reducer.fit_transform(batchData)
                X = embeddingUMAP[:neighborEnd,:]
                if j ==2:
                    for weights in ['uniform', 'distance']:
                        # we create an instance of Neighbours Classifier and fit the data.
                        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                        clf.fit(X, y[i])
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




    for j in range(2,
