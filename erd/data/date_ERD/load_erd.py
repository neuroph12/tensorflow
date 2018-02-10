from scipy import *
from scipy.io import loadmat
import matplotlib.pyplot as pyplot
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FastICA

class Load_erd():
  def __init__(self, data_name='data_151116_zr01.mat'):
    data = loadmat(data_name)
    self.eeg = data['yout'][:, [1,3,9,10,12]].astype(np.float32)
    self.erd = data['yout'][:, 20:24].astype(np.float32)
    label_walk = data['yout'][:, 18]
    label_rest = 1 - label_walk
    self.label = np.c_[label_walk, label_rest].astype(np.float32)

  def get_data1d(self):
    return self.erd, self.label

  def get_data2d(self, seq_len):

    def trainspose2D(train_data, label_data):
      num_batch = len(train_data) - seq_len + 1
      x = np.zeros((num_batch, seq_len, self.erd.shape[1]))
      for start in range(len(train_data) - seq_len + 1):
        x[start, :, :] = train_data[start: start + seq_len]
      label_data = label_data[:num_batch]
      return x, label_data

    self.erd2d, self.label2d = trainspose2D(self.erd, self.label)
    return self.erd2d, self.label2d

  def pca(self, whiten = False):
    pca = PCA(n_components = 4, whiten = whiten)
    self.erd = pca.fit_transform(self.erd)

  def ica(self, whiten = False):
    ica = FastICA(n_components = 4, whiten = whiten)
    self.erd = ica.fit_transform(self.erd)
