import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import six
import tensorflow as tf
from edward.models import Categorical, Dirichlet, Empirical, InverseGamma, \
    MultivariateNormalDiag, Normal, ParamMixture
from mixed_gauss import Mixed_gauss_model
from loaddata import Load_data


data = Load_data()
train, test, label = data.get_data1d()
label = label[:, 0]

N = train.shape[0]
K = 2  # number of components
D = train.shape[1]  # dimensionality of data
T = 3  # number of mcmc samples
ed.set_seed(42)

model = Mixed_gauss_model(N, K, D, T)
model.fit(train)
model.clustering(train)
model.plot_clusters(train, label, axis = [0, 1])
