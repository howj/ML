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

# 1.3 Stochastic Gradient Descent

# 1.3.1.b

def SGDfunctionF(X, Y, Yhat, lamb, W, n, randomPoint):
    constant = -1.0
    return constant * (np.dot((np.transpose(np.subtract(Y[randomPoint],
                                                        Yhat[randomPoint]))), X[randomPoint])) + lamb * W

# Gradient Descent for 1.2.3
def SGDtrain(K, eta, X, Y, lamb, avgSqErr, classError, name, etaDecrease):
    minAvg = 100.0
    minMCE = 100.0
    Wlist = []
    W = np.zeros(X.shape[1], )
    for i in range(1, K + 1):
        Yhat = computeYhat(X, W)
        # compute gradient with one random point
        n = np.shape(X)
        randomPoint = random.randint(0, n[0] - 1)
        gradient = SGDfunctionF(X, Y, Yhat, lamb, W, n, randomPoint)
        W = W - (eta * gradient)

        error1 = compute_class_error(Y, Yhat, 0.5)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
        # decrease eta every single iteration?
        eta -= etaDecrease

        # # decrease eta every 500 updates
        # if i % 500 == 0:
        #     eta -= etaDecrease
        minAvg = min(minAvg, error2)
        minMCE = min(minMCE, error1)
        Wlist.append(W)

    return np.array(Wlist), avgSqErr, classError

# for test and devPercent
def SGD(Wlist, K, eta, X, Y, lamb, avgSqErr, classError, name, etaDecrease):
    minAvg = 100.0
    minMCE = 100.0
    for i in range(1, K + 1):
        W = Wlist[i-1]
        Yhat = computeYhat(X, W)
        error1 = compute_class_error(Y, Yhat, 0.5)
        error2 = compute_average_squared_error(X, Y, W)
        avgSqErr.append(error2)
        classError.append(error1)
        # print "Dataset:", name, ", iter # ", i, ", avg squared error : ", error2, "class error: ", error1
        minAvg = min(minAvg, error2)
        minMCE = min(minMCE, error1)
    print minAvg, minMCE
    return avgSqErr, classError

iters = 5000 # 0.04, @ 0.00001, 4000 steps = stepsize 0
# etaDecrease = 0.000001
iStepSize = 0.005
etaDecrease = 0.00000001
SGDlamb = 0.0001
# Train
WtrainGD, trainAvgSqErr, trainClassError = SGDtrain(iters, iStepSize,
                                                    Xtrain, Ytrain, SGDlamb, [], [], "Train", etaDecrease)
# Test
testAvgSqErr, testClassError = SGD(WtrainGD, iters, iStepSize, Xtest, Ytest, SGDlamb, [], [], "Test", etaDecrease)
# Dev
devAvgSqErr, devClassError = SGD(WtrainGD, iters, iStepSize, Xdev, Ydev, SGDlamb, [], [], "Dev", etaDecrease)

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