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

# A3 1.2: Linear Regression Using Gradient Descent

# From 1.2.2
def functionF(X, Y, Yhat, lamb, W):
    n = np.shape(X)
    constant = -1.0 / n[0]
    return constant * (np.dot((np.transpose(np.subtract(Y, Yhat))), X)) + lamb * W

# Gradient Descent for 1.2.3
def testGradientDescent(Wlist, K, eta, X, Y, lamb, avgSqErr, classError, name):
    W = np.zeros(X.shape[1], )
    minAvg = 100.0
    minMCE = 100.0
    for i in range(1, K + 1):
        Yhat = computeYhat(X, W)
        gradient = functionF(X, Y, Yhat, lamb, W)
        W = W - (eta * gradient)
        error1 = compute_class_error(Y, Yhat, 0.5)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
        Wlist.append(W)

        # compare lowest error
        minAvg = min(minAvg, error2)
        minMCE = min(minMCE, error1)
    print "Dataset:", name, ", min AVG error: ", minAvg, ", min MCE error: ", minMCE
    return np.array(Wlist), W, avgSqErr, classError

def gradientDescent(Wlist, K, eta, X, Y, lamb, avgSqErr, classError, name):
    minAvg = 100.0
    minMCE = 100.0
    for i in range(1, K + 1):
        W = Wlist[i-1]
        Yhat = computeYhat(X, W)
        error1 = compute_class_error(Y, Yhat, 0.5)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        minAvg = min(minAvg, error2)
        minMCE = min(minMCE, error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
    print "Dataset:", name, ", min AVG error: ", minAvg, ", min MCE error: ", minMCE
    return avgSqErr, classError


# 1.2.3: Run gradient descent

iters = 15000 # 3000
lamb = 0.005

# Train
Wlist, WtrainGD, trainAvgSqErr, trainClassError = \
    testGradientDescent([], iters, 0.04, Xtrain, Ytrain, lamb, [], [], "Train")
# Test
testAvgSqErr, testClassError = gradientDescent(Wlist, iters, 0.04, Xtest, Ytest, lamb, [], [], "Test")
# Dev
devAvgSqErr, devClassError = gradientDescent(Wlist, iters, 0.04, Xdev, Ydev, lamb, [], [], "Dev")

# 1.2.3.b: plot iter vs. avg sq error for gradient descent
fig = plt.figure()
train, = plt.plot(np.arange(10, iters), trainAvgSqErr[10:], label="Train error")
test, = plt.plot(np.arange(10, iters), testAvgSqErr[10:], label="Test error")
dev, = plt.plot(np.arange(10, iters), devAvgSqErr[10:], label="Dev error")

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