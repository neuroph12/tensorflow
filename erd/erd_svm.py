import sklearn
import numpy as np
import os, sys
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from data.date_ERD.load_erd import Load_erd
from sklearn import svm
from sklearn.decomposition import PCA, FastICA





'''
データの生成
'''
os.chdir('data/date_ERD')

train_data = Load_erd(data_name='data_151116_zr01.mat')
train, train_label = train_data.get_data1d()
train_label = train_label[:, 0]

test_data = Load_erd(data_name='data_151116_zr02.mat')
test, test_label = test_data.get_data1d()
test_label = test_label[:, 0]

pca = PCA(n_components = 4, whiten = True)
train = pca.fit(train)

plt.scatter(train[:,0], train[:,1], c=train_label)
plt.show()
# clf = svm.SVC()
# clf.fit(train, train_label)
