import numpy as np
from scipy import *
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FastICA
## data import

class Load_data():

    def __init__(self, train_mat, test_mat, train_label_mat, test_label_mat):


        def load_data(matfile):
            data = loadmat(matfile)
            return data['eeg_foot_5ch'].astype(np.float32)

        self.train = load_data(train_mat)
        self.test = load_data(test_mat)

        def load_label(matfile):
            data = loadmat(matfile)
            label = data['C'].astype(np.float32)
            for i in range(len(label)):
                if label[i] < 0:
                    label[i] = 0
                if label[i] > 1:
                    label[i] = 0
            return np.c_[label, 1 - label]

        self.train_label = load_label(train_label_mat)
        self.test_label = load_label(test_label_mat)

    def pca(self, whiten = True):
        pca = PCA(n_components = 5, whiten = whiten)
        self.train = pca.fit_transform(self.train)

    def ica(self, whiten = True):
        ica = FastICA(n_components = 5, whiten = whiten)
        self.train = ica.fit_transform(self.train)

    def get_data2d(self, seq_len):

        def trainspose2D(train_data, label_data):
            num_batch = len(train_data) - seq_len + 1
            x = np.zeros((num_batch, seq_len, 5))
            for start in range(len(train_data) - seq_len + 1):
                x[start, :, :] = train_data[start: start + seq_len]
            label_data = label_data[:num_batch]
            return x, label_data

        self.train, self.train_label = trainspose2D(self.train, self.train_label)
        self.test, self.test_label = trainspose2D(self.test, self.test_label)
        return self.train, self.train_label, self.test, self.test_label

    def get_data1d(self):
        return self.train, self.test, self.label

    def split(self, rate = 0.7):
        cut_idx = int(np.ceil(rate * len(self.train)))
        train, label = shuffle(self.train, self.label)
        self.train = train[:cut_idx]
        self.train_label = label[:cut_idx]
        self.val = train[cut_idx:]
        self.val_label = label[cut_idx:]
        return self.train, self.train_label, self.val, self.val_label



def make_data(train_mat="train_foot2.mat",
              test_mat="test_foot2.mat",
              train_label_mat="label_foot.mat",
              test_label_mat="label_foot.mat",
              seq_len=64):
    data = Load_data(train_mat=train_mat,
                     test_mat=test_mat,
                     train_label_mat=train_label_mat,
                     test_label_mat=test_label_mat)
    train, train_label, test, test_label = data.get_data2d(seq_len)
    return train, train_label, test, test_label
