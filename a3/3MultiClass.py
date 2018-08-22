import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
import random

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




# Closed form, from A3 1.1

# lambda and identity matrix
lamb = 1
idMa = np.identity(784) # identity matrix d x d

# Compute the labels

Ytrain2 = computeLabel(Ytrain, np.shape(Xtrain))
Ytest2 = computeLabel(Ytest, np.shape(Xtest))
Ydev2 = computeLabel(Ydev, np.shape(Xdev))

# Get the W's
W = computeW(Xtrain, Ytrain2)

# Get the Yhats
yHatTrain = computeYhat(Xtrain, W)
yHatTest = computeYhat(Xtest, W)
yHatDev = computeYhat(Xdev, W)

# # Xtrain
# print "----------------------------------------------------------"
# print "Training Misclassification Error: ", compute_class_error(Ytrain2, yHatTrain)
# print "Training Average Squared Error: ", compute_average_squared_error(Xtrain, Ytrain2, W)
#
# # Xtest
# print "----------------------------------------------------------"
# print "Test Misclassification Error: ", compute_class_error(Ytest2, yHatTest)
# print "Test Average Squared Error: ", compute_average_squared_error(Xtest, Ytest2, W)
#
# # Xdev
# print "----------------------------------------------------------"
# print "Dev Misclassificationclear Error: ", compute_class_error(Ydev2, yHatDev)
# print "Dev Average Squared Error: ", compute_average_squared_error(Xdev, Ydev2, W)
# print "----------------------------------------------------------"

# Gradient Descent (Slow as heck!)

# From 1.2.2
def functionF(X, Y, Yhat, lamb, W):
    n = np.shape(X)
    constant = -1.0 / n[0]
    # print np.shape(np.transpose(X))
    return constant * (np.dot(np.transpose(X), np.subtract(Y, Yhat))) + lamb * W

# Gradient Descent for 1.3
def gradientDescent(K, eta, X, Y, lamb, avgSqErr, classError, name):
    Wshape = (X.shape[1], 10)
    W = np.zeros(Wshape)
    for i in range(1, K + 1):
        Yhat = computeYhat(X, W)
        gradient = functionF(X, Y, Yhat, lamb, W)
        W = W - (eta * gradient)

        # print np.shape(X)
        # print np.shape(Y), np.shape(Yhat)

        error1 = compute_class_error(Y, Yhat)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1

    return W, avgSqErr, classError

# 1.2.3: Run gradient descent

iters = 800 # 3000

n = np.shape(Xtrain)
Ytrain2 = computeLabel(Ytrain, n)

WtrainGD, trainAvgSqErr, trainClassError = gradientDescent(iters, 0.04, Xtrain, Ytrain2, 0.1, [], [], "Train")
# Test
WtestGD, testAvgSqErr, testClassError = gradientDescent(iters, 0.04, Xtest, Ytest, 0.1, [], [], "Test")
# Dev
WdevGD, devAvgSqErr, devClassError = gradientDescent(iters, 0.04, Xdev, Ydev, 0.1, [], [], "Dev")

# 1.2.3.b: plot iter vs. avg sq error for gradient descent

fig = plt.figure()
train, = plt.plot(np.arange(20, iters), trainAvgSqErr[20:], label="Train error")
test, = plt.plot(np.arange(20, iters), testAvgSqErr[20:], label="Test error")
dev, = plt.plot(np.arange(20, iters), devAvgSqErr[20:], label="Dev error")
# with gzip.open("mnist.pkl.gz") as f:
#     data = pickle.load(f)
# Xtrain, Ytrain, Xtest, Ytest, Xdev, Ydev = data[b"Xtrain"], data[b"Ytrain"], \
#                                            data[b"Xtest"], data[b"Ytest"], data[b"Xdev"], data[b"Ydev"]

plt.legend()

fig.suptitle("Average Squared Error")
plt.xlabel("Iter #")
plt.ylabel("AvgSqErr")
plt.show()

# 1.2.3.c: plot iter vs. missclassification error for gradient descent

fig = plt.figure()
train, = plt.plot(np.arange(20, iters), trainClassError[20:], label="Train error")
test, = plt.plot(np.arange(20, iters), testClassError[20:], label="Test error")
dev, = plt.plot(np.arange(20, iters), devClassError[20:], label="Dev error")

plt.legend()

fig.suptitle("Misclassification Error")
plt.xlabel("Iter #")
plt.ylabel("Misclassification Error")
plt.show()