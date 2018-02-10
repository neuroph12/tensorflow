
import edward as ed
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from edward.models import Normal, Bernoulli, OneHotCategorical
from data.date_ERD.load_erd import Load_erd
from scipy.io import loadmat
import os, sys
sys.path.append(os.pardir)
from module.utility.history import History, Real_time_plot, EarlyStopping

## data import
dataset = loadmat('train_data_CNN2.mat')
train = dataset['train_data']
train = np.array(train)
train = train.astype(np.float32)
train = train.reshape((len(train), 28, 28, 1))

dataset = loadmat('test_data_CNN2.mat')
test = dataset['train_data']
test = np.array(test)
test = test.astype(np.float32)
test = test.reshape((len(train), 28, 28, 1))

dataset = loadmat('label_data_CNN.mat')
label = dataset['label_data']
label = np.array(label)
label = label.astype(np.int32)
label2 = 1 - label
label = np.c_[label2,label]

train_label = label
test_label = label


# model
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
t = tf.placeholder(tf.float32, [None, 2])

with tf.name_scope('Model'):
    with tf.name_scope('conv1'):
        W_conv1 = Normal(loc=tf.zeros([5, 5, 1, 16]),
                        scale=0.1*tf.ones([5, 5, 1, 16]))
        b_conv1 = Normal(loc=tf.zeros([16]), scale=tf.ones([16]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1,
                        strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        W_conv2 = Normal(loc=tf.zeros([5, 5, 16, 32]),
                        scale=0.1*tf.ones([5, 5, 16, 32]))
        b_conv2 = Normal(loc=tf.zeros([32]), scale=tf.ones([32]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2,
                        strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])

    with tf.name_scope('dense1'):
        W_fc1 = Normal(loc=tf.zeros([7*7*32, 512]),
                        scale=0.1*tf.ones([7*7*32, 512]))
        b_fc1 = Normal(loc=tf.zeros([512]), scale=3.0*tf.ones([512]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dense2'):
        W_fc2 = Normal(loc=tf.zeros([512, 2]),
                        scale=0.1*tf.ones([512, 2]))
        b_fc2 = Normal(loc=tf.zeros([2]), scale=3.0*tf.ones([2]))
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    with tf.name_scope('prob'):
        y = OneHotCategorical(h_fc2)

with tf.name_scope('posterior'):
    with tf.name_scope('qW_conv1'):
        qW_conv1 = Normal(loc=tf.Variable(
                            tf.random_normal([5, 5, 1, 16]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([5, 5, 1, 16]), name='scale')))
    with tf.name_scope('qb_conv1'):
        qb_conv1 = Normal(loc=tf.Variable(
                            tf.random_normal([16]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([16]), name='scale')))
    with tf.name_scope('qW_conv2'):
        qW_conv2 = Normal(loc=tf.Variable(
                            tf.random_normal([5, 5, 16, 32]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([5, 5, 16, 32]), name='scale')))
    with tf.name_scope('qb_conv2'):
        qb_conv2 = Normal(loc=tf.Variable(
                            tf.random_normal([32]), name='loc'),
                            scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([32]), name='scale')))
    with tf.name_scope('qW_fc1'):
        qW_fc1 = Normal(loc=tf.Variable(
                            tf.random_normal([7*7*32, 512]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([7*7*32, 512]), name='scale')))
    with tf.name_scope('qb_fc1'):
        qb_fc1 = Normal(loc=tf.Variable(
                            tf.random_normal([512]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([512]), name='scale')))
    with tf.name_scope('qW_fc2'):
        qW_fc2 = Normal(loc=tf.Variable(
                            tf.random_normal([512, 2]), name='loc'),
                          scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([512, 2]), name='scale')))
    with tf.name_scope('qb_fc2'):
        qb_fc2 = Normal(loc=tf.Variable(
                            tf.random_normal([2]), name='loc'),
                            scale=tf.nn.softplus(tf.Variable(
                            tf.random_normal([2]), name='scale')))

inference = ed.KLqp({W_conv1: qW_conv1, b_conv1: qb_conv1,
                     W_conv2: qW_conv2, b_conv2: qb_conv2,
                     W_fc1: qW_fc1, b_fc1: qb_fc1,
                     W_fc2: qW_fc2, b_fc2: qb_fc2},
                     data={x: train, t: label})

inference.initialize(n_samples=5, n_print=10, n_iter=1000, logdir='log')
tf.global_variables_initializer().run()
ax = Real_time_plot()

with tf.name_scope('pridictor'):
    # data = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
    conv1 = tf.nn.relu(tf.nn.conv2d(train, qW_conv1.sample(),
                        strides=[1, 1, 1, 1], padding='SAME')
                        + qb_conv1.sample())
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, qW_conv2.sample(),
                        strides=[1, 1, 1, 1], padding='SAME')
                        + qb_conv2.sample())
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

    pool2_flat = tf.reshape(pool2, [-1, 7*7*32])

    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W_fc1.sample()) + b_fc1.sample())
    fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2.sample()) + b_fc2.sample())
    prob = tf.nn.softmax(fc2)
    label = tf.one_hot(tf.argmax(prob, axis=1), depth=2)
    predict = tf.argmax(tf.reduce_mean(
                tf.stack([label for _ in range(10)]), axis=0), axis=1)


print('start learning')
for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

  if t % inference.n_print == 0:
    #   print(predict.eval().shape)
    ax.plot(predict.eval())


#inference.run(logdir='log')
