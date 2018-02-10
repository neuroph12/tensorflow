import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy import *
from scipy.io import loadmat

#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



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
label1 = np.array(label).astype(np.float32)
label2 = 1 - label1
label = np.c_[label2, label1]

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
t = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




#first layer
W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#second layer
W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#dense layer
W_fc1 = weight_variable([7 * 7 * 16, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#setting optimizer
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y_conv) + 0.01 * tf.reduce_sum( tf.abs( W_conv1 )) + 0.01 * tf.reduce_sum( tf.abs( W_conv2 )))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#learning loop
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: train, t: label, keep_prob: 1.0})
      test_accuracy = accuracy.eval(feed_dict={x: test, t: label, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      print('step %d, test accuracy %g' % (i, test_accuracy))
    train_step.run(feed_dict={x: train, t: label, keep_prob: 0.5})
      
