import gzip, pickle
with gzip.open('mnist.pkl.gz') as f:
    train_set, valid_set, test_set = pickle.load(f)

