import edward as ed
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from edward.models import Normal, Bernoulli, OneHotCategorical
from data.date_ERD.load_erd import Load_erd
import os, sys
sys.path.append(os.pardir)
from module.utility.history import History, Real_time_plot, EarlyStopping

os.chdir('data/date_ERD')

train_data = Load_erd(data_name='data_151116_zr02_s.mat')
train, label = train_data.get_data1d()

test_data = Load_erd(data_name='data_151116_zr01_s.mat')
test, _ = test_data.get_data1d()


def neural_network(X):
  h = tf.nn.relu(tf.matmul(X, W_0) + b_0)
  h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
  h = tf.nn.sigmoid(tf.matmul(h, W_2) + b_2)
  return h

ed.set_seed(42)

N = train.shape[0]  # number of data points
D = train.shape[1]  # number of features

# MODEL
# with tf.name_scope("model"):
#   W_0 = Normal(loc=tf.zeros([D, 30]), scale=tf.ones([D, 30]), name="W_0")
#   W_1 = Normal(loc=tf.zeros([30, 20]), scale=tf.ones([30, 20]), name="W_1")
#   W_2 = Normal(loc=tf.zeros([20, 1]), scale=tf.ones([20, 1]), name="W_2")
#   b_0 = Normal(loc=tf.zeros(30), scale=tf.ones(30), name="b_0")
#   b_1 = Normal(loc=tf.zeros(20), scale=tf.ones(20), name="b_1")
#   b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name="b_2")
#
#   X = tf.placeholder(tf.float32, [N, D], name="X")
#   prob = neural_network(X)
#   y = Bernoulli(logits=prob, name="y")
#
# # INFERENCE
# with tf.name_scope("posterior"):
#   with tf.name_scope("qW_0"):
#     qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, 30]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([D, 30]), name="scale")))
#   with tf.name_scope("qb_0"):
#     qb_0 = Normal(loc=tf.Variable(tf.random_normal([30]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([30]), name="scale")))
#
#   with tf.name_scope("qW_1"):
#     qW_1 = Normal(loc=tf.Variable(tf.random_normal([30, 20]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([30, 20]), name="scale")))
#   with tf.name_scope("qb_1"):
#     qb_1 = Normal(loc=tf.Variable(tf.random_normal([20]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([20]), name="scale")))
#
#   with tf.name_scope("qW_2"):
#     qW_2 = Normal(loc=tf.Variable(tf.random_normal([20, 1]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([20, 1]), name="scale")))
#   with tf.name_scope("qb_2"):
#     qb_2 = Normal(loc=tf.Variable(tf.random_normal([1]), name="loc"),
#                   scale=tf.nn.softplus(
#                       tf.Variable(tf.random_normal([1]), name="scale")))
#
# inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
#                      W_1: qW_1, b_1: qb_1,
#                      W_2: qW_2, b_2: qb_2}, data={X: train, y: label[:,1].reshape(-1, 1)})
#inference.run(logdir='log')

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


inference.initialize(n_print=100, n_iter=10000, n_samples=5)

tf.global_variables_initializer().run()
ax = Real_time_plot()


def fwd_infer(x):
  h = tf.nn.relu(ed.dot(x, qW_0.sample()) + qb_0.sample())
  h = tf.nn.relu(ed.dot(h, qW_1.sample()) + qb_1.sample())
  h = tf.nn.sigmoid(ed.dot(h, qW_2.sample()) + qb_2.sample())
  return h

# Build samples from inferred posterior.

print('start learning')
for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    predict = tf.round(tf.sigmoid(ed.dot(train, qw.sample()) + qb.sample()))
    # predict = tf.round(fwd_infer(train))
    correct_prediction = tf.equal(predict, label[:,1])
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_predict = tf.round(tf.sigmoid(ed.dot(test, qw.sample()) + qb.sample()))
    # val_predict = tf.round(fwd_infer(test))
    val_correct_prediction = tf.equal(val_predict, label[:,1])
    val_accuracy = tf.reduce_mean(tf.cast(val_correct_prediction, tf.float32))
    print('\n  \n training_accuracy : {}'.format(accuracy.eval()))
    print(' validation_accuracy : {} \n \n '.format(val_accuracy.eval()))

    ax.plot(predict.eval())
