#Sourced from https://towardsdatascience.com/ml-from-scratch-k-nearest-neighbors-classifier-3fc51438346b
import numpy as np


def euclidean_dist(pointA, pointB):
    """Calculates the euclidean distance between two vectors (numpy arrays).
    Args-
        pointA- First vector
        pointB- Second vector
    Returns-
        distance- Euclidean distance between A and B
    """
    distance = np.square(pointA - pointB) # (ai-bi)**2 for every point in the vectors
    distance = np.sum(distance) # adds all values
    distance = np.sqrt(distance) 
    return distance

def distance_from_all_training(test_point,X_train):
    """Calculates euclidean distance of test point from all the points in the training dataset
    Args-
        test_point- Data point from test set
    Returns-
        dist_array- Array holding distance values for all training data points
    """
    dist_array = np.array([])
    for train_point in X_train:
        dist = euclidean_dist(test_point, train_point)
        dist_array = np.append(dist_array, dist)
    return dist_array


def KNNClassifier(train_features, train_target, test_features, k = 5):
    """Performs KNN classification on the test feature set.
    Args-
        train_features- This denotes the feature set of the training data
        train_target- Target lables of the training data
        test_features- Feature set of the test data; assumed unlabeled
        k (default = 5)- Number of closest neighboring training data points to be considered
    Returns-
        predictions- Array of target predictions for each test data instance
    """
    predictions = np.array([])
    train_target = train_target.reshape(-1,1)
    for test_point in test_features: # iterating through every test data point
        dist_array = distance_from_all_training(test_point,train_features).reshape(-1,1) # calculating distance from every training data instance
        neighbors = np.concatenate((dist_array, train_target), axis = 1)
        neighbors_sorted = neighbors[neighbors[:, 0].argsort()] # sorts training points on the basis of distance
        k_neighbors = neighbors_sorted[:k] # selects k-nearest neighbors
        frequency = np.unique(k_neighbors[:, 1], return_counts=True)
        target_class = frequency[0][frequency[1].argmax()] # selects label with highest frequency
        predictions = np.append(predictions, target_class)

    return predictions


def accuracy(y_test, y_preds):
    """Calculates inference accuracy of the model.

    Args-
        y_test- Original target labels of the test set
        y_preds- Predicted target lables
    Returns-
        acc
    """
    total_correct = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == int(y_preds[i]):
            total_correct += 1
    acc = total_correct/len(y_test)
    return acc
