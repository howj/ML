import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
import random

# initial setup stuff
with gzip.open("mnist_2_vs_9.gz") as f:
    data = pickle.load(f)
Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], \
                                           data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]

# Set the last column to be all ones
Xtrain[:,783] = 1
Xtest[:,783] = 1
Xdev[:,783] = 1

# Figuring out the percent error, Sham's function from Canvas
def compute_class_error(Y,Yhat,b):
    Yhat_labels=(Yhat-b)>=0
    errors = np.abs(Yhat_labels-Y)
    return 100*sum(errors)/(Yhat.shape[0]*1.0)

# Computes the average squared error
def compute_average_squared_error(X, Y, W):
    n = np.shape(X)
    constant = 1.0 / n[0]
    return 0.5 * constant * (np.linalg.norm(Y - np.dot(X, W)) ** 2)

# Computes the W
def computeW(X, Y):
    n = np.shape(X)
    constant = 1.0 / n[0]
    w1 = np.linalg.inv(constant * np.dot(np.transpose(X), X) + lamb * idMa)
    w2 = (constant * (np.dot(np.transpose(X), Y)))
    return np.dot(w1, w2)

# Computes Yhats
def computeYhat(X, W):
    n = np.shape(X)
    yhat = np.empty(((n[0]), ))
    for index in range(n[0]):
        yhat[index] = np.dot(X[index], W)
    return yhat

# A3 1.1

# lambda and identity matrix
lamb = 1
idMa = np.identity(784) # identity matrix d x d

# Get the W's
W = computeW(Xtrain, Ytrain)

# Get the Yhats
yHatTrain = computeYhat(Xtrain, W)
yHatTest = computeYhat(Xtest, W)
yHatDev = computeYhat(Xdev, W)

# Xtrain
print "----------------------------------------------------------"
print "Training Misclassification Error: ", compute_class_error(Ytrain, yHatTrain, 0.5)
print "Training Average Squared Error: ", compute_average_squared_error(Xtrain, Ytrain, W)

# Xtest
print "----------------------------------------------------------"
print "Test Misclassification Error: ", compute_class_error(Ytest, yHatTest, 0.5)
print "Test Average Squared Error: ", compute_average_squared_error(Xtest, Ytest, W)

# Xdev
print "----------------------------------------------------------"
print "Dev Misclassification Error: ", compute_class_error(Ydev, yHatDev, 0.5)
print "Dev Average Squared Error: ", compute_average_squared_error(Xdev, Ydev, W)
print "----------------------------------------------------------"
