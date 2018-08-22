import numpy as np
import gzip, pickle
import matplotlib.pyplot as plt
import random
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import Image
# import PIL
from torch.utils.data import DataLoader

# initial setup stuff

with gzip.open('mnist.pkl.gz') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')

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

# visualizes the weights
def visualizeInputWeights(model):
    nodes = np.random.choice(np.random.choice(100), size = 8)
    initialW = np.zeros((8, 784))
    modelW = model[0].weight.data
    for idx in range (8):
        initialW[idx] = modelW[nodes[idx]]

    vizList = []
    for w in initialW:
        # Reshape to 28 x 28
        vizList.append(stand(w).reshape(28, 28))

    fig = plt.figure(figsize=(8, 8))
    c = 4
    r = 2
    k = 0
    for j in range(1, c * r + 1):
        viz = vizList[k]
        fig.add_subplot(r, c, j)
        plt.imshow(viz, cmap='gray')
        k += 1
    plt.show()

# Sigmoid function
def sigmoid(a):
    return 1 / (1 + math.exp(-a))

# lambda and identity matrix
lamb = 1
idMa = np.identity(784) # identity matrix d x d

# Compute the labels

Ytrain2 = computeLabel(Ytrain, np.shape(Xtrain))
Ytest2 = computeLabel(Ytest, np.shape(Xtest))
Ydev2 = computeLabel(Ydev, np.shape(Xdev))

# Weights init from the XOR section slides
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # initialize the weight tensor, here we use a normal distribution
            m.weight.data.normal_(0, 1)

# Model creation, two layer NN with one hidden layer
def createModel(input, hiddenNodes, output):
    return nn.Sequential(nn.Linear(input, hiddenNodes), nn.Sigmoid(), nn.Linear(hiddenNodes, output), nn.Sigmoid())

def createModelReLu(input, hiddenNodes, output):
    return nn.Sequential(nn.Linear(input, hiddenNodes), nn.ReLU(), nn.Linear(hiddenNodes, output), nn.ReLU())

# Inspired from the section slides
def SGDTrain(input_size, hiddenNodes, output_size, num_epochs, X, Y, batch_size,
             Xtest, Ytest, Xdev, Ydev, learning_rate):
    minTrainMSE = 100.0
    minTrainMCE = 100.0
    minTestMSE = 100.0
    minTestMCE = 100.0
    minDevMSE = 100.0
    minDevMCE = 100.0

    my_model = createModel(input_size, hiddenNodes, output_size)
    # my_model = createModelReLu(input_size, hiddenNodes, output_size) # strange, doesnt change
    # weights_init(my_model)
    opt = optim.SGD(my_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay = 0.00001)

    TrainMSE = []
    TrainMCE = []
    TestMSE = []
    TestMCE = []
    DevMSE = []
    DevMCE = []
    EPOCHS = []
    count = 0.5

    for idx in range(1, num_epochs + 1):
        # Shuffle the X, Y
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        X = X[s]
        Y = Y[s]
        x_var = Variable(torch.Tensor(X[0:batch_size]))
        y_var = Variable(torch.Tensor(Y[0:batch_size]))
        my_model.zero_grad()

        # Get the X, Y's for test and dev as Variables
        x_var_test = Variable(torch.Tensor(Xtest))
        y_var_test = Variable(torch.Tensor(Ytest))
        x_var_dev = Variable(torch.Tensor(Xdev))
        y_var_dev = Variable(torch.Tensor(Ydev))

        # compute yhats
        y_hat = my_model(x_var)
        YhatTest = my_model(x_var_test)
        YhatDev = my_model(x_var_dev)

        # get the losses (MSE)
        loss = F.mse_loss(y_hat, y_var)  # this is the mean squared error
        lossTest = F.mse_loss(YhatTest, y_var_test)
        lossDev = F.mse_loss(YhatDev, y_var_dev)
        # get the min
        minTrainMSE = min(minTrainMSE, loss.data[0])
        minTestMSE = min(minTestMSE, lossTest.data[0])
        minDevMSE = min(minDevMSE, lossDev.data[0])

        # get the losses (MCE)
        trainMCE = compute_class_error(y_var.data.numpy(), y_hat.data.numpy())
        testMCE = compute_class_error(y_var_test.data.numpy(), YhatTest.data.numpy())
        devMCE = compute_class_error(y_var_dev.data.numpy(), YhatDev.data.numpy())
        minTrainMCE = min(minTrainMCE, trainMCE)
        minTestMCE = min(minTestMCE, testMCE)
        minDevMCE = min(minDevMCE, devMCE)

        loss.backward()  # it computes the gradient for us!

        print (batch_size * idx, "TrainMSE:", loss.data[0], ", TrainMCE:", trainMCE, ", DevMSE:",\
            lossDev.data[0], ", DevMCE:", devMCE)

        opt.step()  # this does the parameter for us!
        # if idx % (25024/batch_size) == 0:
        if (idx * batch_size) % 25000 == 0:
            print ("Epoch #:", count, "TrainMSE:", loss.data[0], "TestMSE:", lossTest.data[0], "DevMSE:", lossDev.data[0])
            print ("          TrainMCE:", trainMCE, "TestMCE:", testMCE, "DevMCE:", devMCE)

            # MSE
            TrainMSE.append(loss.data[0])
            TestMSE.append(lossTest.data[0])
            DevMSE.append(lossDev.data[0])

            # MCE
            TrainMCE.append(trainMCE)
            TestMCE.append(testMCE)
            DevMCE.append(devMCE)

            # Other
            EPOCHS.append(count)
            count += 0.5
    # Return our W, v, MSE, MCE
    visualizeInputWeights(my_model)
    print ("Learning rate:", learning_rate)
    print ("Min Train MSE:", minTrainMSE, ", Min Test MSE:", minTestMSE, ", Min Dev MSE:", minDevMSE)
    print ("Min Train MCE:", minTrainMCE, ", Min Test MCE:", minTestMCE, ", Min Dev MCE:", minDevMCE)
    return EPOCHS, TrainMSE, TrainMCE, TestMSE, TestMCE, DevMSE, DevMCE

# 3.1 Try to learn a one hidden layer network

batch_size = 50
iters = 10000 # 10000 = 10
learningRate = 0.20
print ("Examining", batch_size * iters, "datapoints")
# Train
Xaxis, TrainMSE, TrainMCE, TestMSE, TestMCE, DevMSE, DevMCE = SGDTrain(784, 100, 10, iters, Xtrain, Ytrain2, batch_size,
                                                                       Xtest, Ytest2, Xdev, Ydev2, learningRate)

########################################################################################################################

# plot iter vs. avg sq error

fig = plt.figure()
train, = plt.plot(Xaxis[0:], TrainMSE[0:], label="Train error")
test, = plt.plot(Xaxis[0:], TestMSE[0:], label="Test error")
dev, = plt.plot(Xaxis[0:], DevMSE[0:], label="Dev error")

plt.legend()

fig.suptitle("Average Squared Error")
plt.xlabel("Effective Iter #")
plt.ylabel("MSE")
plt.show()

# plot iter vs. misclassification error

fig = plt.figure()
train, = plt.plot(Xaxis[2:], TrainMCE[2:], label="Train error")
test, = plt.plot(Xaxis[2:], TestMCE[2:], label="Test error")
dev, = plt.plot(Xaxis[2:], DevMCE[2:], label="Dev error")

plt.legend()

fig.suptitle("Misclassification Error")
plt.xlabel("Effective Iter #")
plt.ylabel("Misclassification Error")
plt.show()