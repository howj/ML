import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

DEVSET = False # set this to True to train

# check for correct amount of args
if not len(sys.argv) == 5:
    print "Error, wrong amount of command line args."
    print "Usage: [training_filename.tsv] [test_filename.tsv] [Maximum epochs] " \
          "[Epoch to start recording data (enter 0 for default)]"
    sys.exit(1)

# arg stuff
filename = sys.argv[1]
filename2 = sys.argv[2]
NUM_EPOCHS = int(sys.argv[3])
START_EPOCHS = int(sys.argv[4])

if NUM_EPOCHS < START_EPOCHS: # check to see if client provided valid epoch values
    print "Error: Maximum epochs must be larger than epoch to start recording data"
    sys.exit(1)

# initialize plotting stuff
x_axis = np.array([i for i in range(START_EPOCHS, NUM_EPOCHS)])
training_error = np.empty(NUM_EPOCHS - START_EPOCHS)
test_error = np.empty(NUM_EPOCHS - START_EPOCHS)

# read the data
df = np.genfromtxt(filename, delimiter='\t')
testdata = np.genfromtxt(filename2, delimiter='\t')
num_rows = df.shape[0]
num_cols = df.shape[1]

if DEVSET: # slice some rows off if DEVSET
    slice = num_rows / 10
    sliced = num_rows - slice
    df = df[:sliced,]
    num_rows = df.shape[0]

# initialize w, b, u, beta, to be 0, c to 1
w = np.zeros(num_cols - 1) # initialize vector
u = np.zeros(num_cols - 1) # initialize vector
b = 0
beta = 0
c = 1

for i in range (NUM_EPOCHS): # run for specified epochs
    np.random.shuffle(testdata) # shuffle rows every epoch
    mistakes = 0
    for rownum in range (num_rows): # for each row in df
        y = df[rownum][0]
        x = np.empty(num_cols - 1) # empty faster than zeros
        for colnum in range(1, num_cols):
            x[colnum - 1] = (df[rownum][colnum])
        a = np.dot(x, w) + b

        # update if incorrect
        if y * a <= 0:
            mistakes += 1
            w = w + (y*x)
            b = b + y
            u = u + (y*c*x) # update cached weights
            beta = beta + (y*c) # update cached bias

        c += 1 # increment counter regardless of update

    epoch_err = float(mistakes) / num_rows

    # update w?
    w = w - ((1/c)*u)
    b = b - ((1/c)*beta)

    # Testing w on the test data

    test_rows = testdata.shape[0]
    test_cols = testdata.shape[1]

    mistakes_test = 0

    for test_rownum in range(test_rows):  # for each row in test_data
        test_y = testdata[test_rownum][0]
        test_x = np.empty(num_cols - 1)  # empty faster than zeros
        for test_colnum in range(1, test_cols):
            test_x[test_colnum - 1] = (testdata[test_rownum][test_colnum])

        test_a = np.dot(test_x, w) + b

        if test_y * test_a <= 0:
            mistakes_test += 1

    test_error_rate = float(mistakes_test) / test_rows

    if (i >= START_EPOCHS):
        training_error[i - START_EPOCHS] = epoch_err
        test_error[i - START_EPOCHS] = test_error_rate
        print "in epoch: ", i, ", training mistakes made: ", mistakes, ", test errs: ", mistakes_test

# plotting stuff

title = "Training, test error rates for ", filename, " from ", START_EPOCHS, " to ", NUM_EPOCHS, "epochs"
plt.title(title)
plt.xlabel('Number of epochs')
plt.ylabel('Error rate')
plt.plot(x_axis, training_error, color='b', label="Training error rate")
plt.plot(x_axis, test_error, color='r', label="Test error rate")
plt.legend()

plt.show()

# print "Done Training! w = ", w - ((1/c) * u), ", b = ", b - ((1/c) * beta)
# print 1/c, u, beta
