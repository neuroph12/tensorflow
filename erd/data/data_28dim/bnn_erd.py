import edward as ed
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from data.data_28dim.loaddata import Load_data
import matplotlib.pyplot as plt
from edward.models import Normal, Bernoulli, OneHotCategorical


data = Load_data()
train, test, label = data.get_data1d()

ed.set_seed(42)

N = train.shape[0]  # number of data points
D = train.shape[1]  # number of features

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros(D), scale=3.0 * tf.ones(D))
b = Normal(loc=tf.zeros([]), scale=3.0 * tf.ones([]))
y = Bernoulli(logits=ed.dot(X, w) + b)

# INFERENCE
qw_loc = tf.Variable(tf.random_normal([D]))
qw_scale = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qb_loc = tf.Variable(tf.random_normal([]) + 10)
qb_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))

qw = Normal(loc=qw_loc, scale=qw_scale)
qb = Normal(loc=qb_loc, scale=qb_scale)

inference = ed.KLqp({w: qw, b: qb}, data={X: train, y: label[:,1]})
inference.initialize(n_print=50, n_iter=1000)

tf.global_variables_initializer().run()

# Set up figure.
fig = plt.figure(figsize=(8, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show()
plt.grid()
plt.ylabel('1:walk , 0:rest')

# Build samples from inferred posterior.
inputs = test

for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    predict = tf.round(tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample()))
    correct_prediction = tf.equal(predict, label[:,1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('\n  \n accuracy : {} \n \n '.format(accuracy.eval()))
    # Plot data and functions
    plt.cla()
    ax.plot(predict.eval())
    ax.set_ylim([-0.5, 1.5])
    plt.grid()
    plt.draw()
    plt.pause(1.0 / 60.0)
