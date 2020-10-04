import numpy as np
import math
from scipy.special import expit
import random

in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 0.001 # Learning rate. You might try different rates (e.g. 0.001, 0.01, 0.1) to maximize the accuracy

def Weight_update(feature, label, weight_i2o):
    predicted_label = np.argmax(np.dot(feature.transpose(), weight_i2o))

    tx = np.zeros(10)
    tx[int(label)] = 1

    yx = np.zeros(10)
    yx[predicted_label] = 1

    gx = np.array([(tx - yx)]) # (tx - yx) with shape (1, 10)
    feature = feature.reshape(in_dim, 1) # feature vector with shape (785, 1)
    weight_i2o += eta * np.dot(feature, gx) # updated weight
    return weight_i2o

def get_predictions(dataset, weight_i2o):
    probabilities = np.dot(dataset, weight_i2o)  # probabilities matrix
    predictions = np.argmax(probabilities, axis=1)  # output array of indexes with maximum probabilities
    return predictions


def train(train_set, labels, weight_i2o):
    #"""
    #Train the perceptron until convergence.
    # Inputs:
        # train_set: training set (ndarray) with shape (number of data points x in_dim)
        # labels: list (or ndarray) of actual labels from training set
        # weight_i2o:
    # Return: the weights for the entire training set
    #"""
    for i in range(0, train_set.shape[0]):
        weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)
    return weight_i2o