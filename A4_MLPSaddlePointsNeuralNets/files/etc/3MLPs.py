import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
import random
import math
import torch.optim as optim

# initial setup stuff

with gzip.open('mnist.pkl.gz') as f:
    train_set, valid_set, test_set = pickle.load(f)

Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = train_set[0], train_set[1], test_set[0], \
                                           test_set[1], valid_set[0], valid_set[1]

# No bias for this one...
# # Set the last column to be all ones
# Xtrain[:,783] = 1
# Xtest[:,783] = 1
# Xdev[:,783] = 1

# Sham's function from canvas for standardizing an image
def stand(x):
    # gives an affine transformation to place x between 0 and 1
    x=x-np.min(x[:])
    x=x/(np.max(x[:])+1e-12)
    return x

# Figuring out misclassification error for multi-class, Sham's other function from Canvas
def compute_class_error(Y,Yhat):
    indsYhat=np.argmax(Yhat,axis=1)
    indsY=np.argmax(Y,axis=1)
    errors = (indsYhat-indsY)!=0
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
    yhat = np.empty(((n[0]), 10))
    for index in range(n[0]):
        yhat[index] = (np.dot(np.transpose(W), X[index]))
    return yhat

# Create the labels for Y in
def computeLabel(Y, n):
    Ylabel = np.empty(((n[0]), 10))
    Yshape = np.shape(Y)
    for i in range (0, Yshape[0]):
        number = Y[i]
        label = np.zeros(10,)
        label[number] = 1
        Ylabel[i] = label
    return np.array(Ylabel)

# Sigmoid function
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

# 3.1 Try to learn a one hidden layer network

# Closed form, from A3 1.1

# lambda and identity matrix
lamb = 1
idMa = np.identity(784) # identity matrix d x d

# Compute the labels

Ytrain2 = computeLabel(Ytrain, np.shape(Xtrain))
Ytest2 = computeLabel(Ytest, np.shape(Xtest))
Ydev2 = computeLabel(Ydev, np.shape(Xdev))

# # Get the W's
# W = computeW(Xtrain, Ytrain2)
#
# # Get the Yhats
# yHatTrain = computeYhat(Xtrain, W)
# yHatTest = computeYhat(Xtest, W)
# yHatDev = computeYhat(Xdev, W)

# SGD from A3
def SGDfunctionF(X, Y, Yhat, lamb, W, n, randomPoint):
    # GD

    # n = np.shape(X)
    # constant = -1.0 / n[0]
    # # print np.shape(np.transpose(X))
    # return constant * (np.dot(np.transpose(X), np.subtract(Y, Yhat))) + lamb * W
    #
    # constant = -1.0
    # # Change to mini-batch

    # term1 = np.subtract(Y[randomPoint], Yhat[randomPoint])
    # print np.shape((np.transpose(term1)))
    # print np.shape(X[randomPoint])
    # return constant * (np.dot((np.transpose(term1)), X[randomPoint])) + lamb * W

        return 1
    # New stuff



# Gradient Descent for 1.2.3
def SGDtrain(K, eta, X, Y, lamb, avgSqErr, classError, name, etaDecrease, mbSize):
    minAvg = 100.0
    minMCE = 100.0
    Wlist = []
    Vlist = []
    D = X.shape[0] # k = 10
    W = np.random.rand(D, 10) # 50,000 x 10, DxK matrix of small random values
    V = np.random.rand(10) # K-vector of small random values
    print (X.shape, Y.shape, W.shape, V.shape)

    for i in range(1, K + 1):

        # New stuff


        G = np.zeros(D, 10)
        g = np.zeros(10)

        for i2 in range (D): # for all x, y in D
            activations = np.empty(10)
            transfers = np.empty(10)
            for i3 in range(10): # for i = 0 to 9
                activations[i3] = np.dot(w[i2], X[i2])
                transfers[i3] = sigmoid(activations[i3])
            Yhat = np.dot(v, transfers)
            error = np.subtract(Y[i2], Yhat)
            g = np.subtract(g, np.dot(error, transfers))
            for i4 in range(10): # for i = 0 to 9
                G[]

        #
        # Yhat = computeYhat(X, W)
        # # compute gradient with one random point
        # n = np.shape(X)
        # randomPoint = random.randint(0, n[0] - 1)
        # gradient = SGDfunctionF(X, Y, Yhat, lamb, W, n, randomPoint)
        # W = W - (eta * gradient)
        #
        # error1 = compute_class_error(Y, Yhat)
        # error2 = compute_average_squared_error(X, Y, W)
        # avgSqErr.append(error2)
        # classError.append(error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
        # # decrease eta every single iteration?
        # eta -= etaDecrease
        #
        # # # decrease eta every 500 updates
        # # if i % 500 == 0:
        # #     eta -= etaDecrease
        # minAvg = min(minAvg, error2)
        # minMCE = min(minMCE, error1)
        Wlist.append(W)

    return np.array(Wlist), avgSqErr, classError

# for test and devPercent
def SGD(Wlist, K, eta, X, Y, lamb, avgSqErr, classError, name, etaDecrease):
    minAvg = 100.0
    minMCE = 100.0
    for i in range(1, K + 1):
        W = Wlist[i-1]
        Yhat = computeYhat(X, W)
        error1 = compute_class_error(Y, Yhat)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
        minAvg = min(minAvg, error2)
        minMCE = min(minMCE, error1)
    print minAvg, minMCE
    return avgSqErr, classError

miniBatchSize = 100
iters = 5000 # 0.04, @ 0.00001, 4000 steps = stepsize 0
# etaDecrease = 0.000001
iStepSize = 0.005
etaDecrease = 0.00000001
SGDlamb = 0.0001
# Train
WtrainGD, trainAvgSqErr, trainClassError = SGDtrain(iters, iStepSize, Xtrain, Ytrain2, SGDlamb,
                                                    [], [], "Train", etaDecrease, miniBatchSize)


# Test
testAvgSqErr, testClassError = SGD(WtrainGD, iters, iStepSize, Xtest, Ytest2, SGDlamb, [], [], "Test", etaDecrease)
# Dev
devAvgSqErr, devClassError = SGD(WtrainGD, iters, iStepSize, Xdev, Ydev2, SGDlamb, [], [], "Dev", etaDecrease)

# 1.3.1.b: plot iter vs. avg sq error for gradient descent

fig = plt.figure()
train, = plt.plot(np.arange(0, iters), trainAvgSqErr[0:], label="Train error")
test, = plt.plot(np.arange(0, iters), testAvgSqErr[0:], label="Test error")
dev, = plt.plot(np.arange(0, iters), devAvgSqErr[0:], label="Dev error")

plt.legend()

fig.suptitle("Average Squared Error")
plt.xlabel("Iter #")
plt.ylabel("AvgSqErr")
plt.show()

# 1.3.1.c: plot iter vs. missclassification error for gradient descent

fig = plt.figure()
train, = plt.plot(np.arange(1500, iters), trainClassError[1500:], label="Train error")
test, = plt.plot(np.arange(1500, iters), testClassError[1500:], label="Test error")
dev, = plt.plot(np.arange(1500, iters), devClassError[1500:], label="Dev error")

plt.legend()

fig.suptitle("Misclassification Error")
plt.xlabel("Iter #")
plt.ylabel("Misclassification Error")
plt.show()