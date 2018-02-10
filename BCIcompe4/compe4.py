import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sns
sys.path.append(os.pardir)
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.contrib.keras as keras
import scipy.signal as signal
import matplotlib.pyplot as plt
os.chdir('/home/seigyo/Documents/tensorflow/BCIcompe4')
## calib_data
## 8秒置きに4秒間 MI
data = loadmat('BCICIV_calib_ds1a.mat')
eeg = data['cnt'].astype(np.float32) * 0.1
pos = data['mrk']['pos'][0][0][0]
label = data['mrk']['y'][0][0][0].astype(np.float32)
fs = data['nfo'][0][0][0][0][0].astype(np.float32)
clab = []
for i in range(59):
  clab.append(data['nfo']['clab'][0][0][0][i][0])
class0 = data['nfo']['classes'][0][0][0][0][0]
class1 = data['nfo']['classes'][0][0][0][1][0]
xpos = data['nfo']['xpos'][0][0]
ypos = data['nfo']['ypos'][0][0]


# test_data = np.sin(2*np.pi*5*t) + 0.1*np.random.randn(len(t))

## butter filters
fn = fs/2
st = 1/fs
mbfa_eeg = []
for i in range(10):
  start = i * 4s
  end = (i + 1) * 4
  cf = [start, end]
  b, a = signal.butter(5, cf/fn, btype='band', analog=False)
  y = signal.lfilter(b, a, eeg)
  # y = signal.lfilter(b, a, test_data)
  mbfa_eeg.append(y)

# ## plot
# dt = 1/fs
# t = np.linspace(1, len(eeg), len(eeg)) * dt - dt
# plt.plot(t[1000:2000], mbfa_eeg[1][1000:2000,40])
# plt.show()

clab.index('C4')
