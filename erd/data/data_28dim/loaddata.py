import numpy as np
from scipy import *
from scipy.io import loadmat

## data import

class Load_data():

    def __init__(self):
        dataset = loadmat('train_data_CNN1.mat')
        train0 = dataset['train_data']
        train0 = np.array(train0)
        train0 = train0.astype(np.float32)
        self.train0 = train0.reshape((len(train0), 28, 28))

        dataset = loadmat('train_data_CNN2.mat')
        train1 = dataset['train_data']
        train1 = np.array(train1)
        train1 = train1.astype(np.float32)
        self.train1 = train1.reshape((len(train1), 28, 28))

        self.train = np.r_[self.train0, self.train1]

        dataset = loadmat('test_data_CNN1.mat')
        test0 = dataset['train_data']
        test0 = np.array(test0)
        test0 = test0.astype(np.float32)
        self.test0 = test0.reshape((len(test0), 28, 28))

        dataset = loadmat('test_data_CNN2.mat')
        test1 = dataset['train_data']
        test1 = np.array(test1)
        test1 = test1.astype(np.float32)
        self.test1 = test1.reshape((len(test1), 28, 28))

        self.test = np.r_[self.test0, self.test1]

        dataset = loadmat('label_data_CNN.mat')
        label = dataset['label_data']
        label = np.array(label)
        label = label.astype(np.int32)
        label2 = 1 - label
        self.label_one = np.c_[label2, label]

        self.label = np.r_[self.label_one, self.label_one]

    def get_data2d(self):
        return self.train, self.test, self.label

    def get_data1d(self):
        self.train = self.train.reshape(self.train.shape[0], -1)
        self.test = self.test.reshape(self.test.shape[0], -1)
        return self.train, self.test, self.label

    def get_one_data2d(self):
        return self.train1, self.test1, self.label_one
