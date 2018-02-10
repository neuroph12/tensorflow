import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import shutil
from scipy.io import loadmat
from load_foot import Load_data, make_data
from module.utility.history import History, Real_time_plot, EarlyStopping
from module.basics.my_layers import bidirectional_LSTM, time_stacked_conv1d, accuracy


'''
モデルハイパーパラメータ
'''
seq_len = 128
freq = 64
lr = 1e-4
epochs = 1000
batch_size = 512
k_prob = 0.6
num_sample = 100

# early_stopping
stop = False
'''
データの生成
'''
os.chdir('data')

train, train_label, test, test_label \
        = make_data(train_mat = "train_foot2.mat",
                    test_mat = "test_foot2.mat",
                    train_label_mat = "label_foot.mat",
                    test_label_mat = "label_foot.mat",
                    seq_len=seq_len)

plot_x = np.r_[train, test]
plot_t = np.r_[train_label, test_label]

N_train = len(train)
n_batches = N_train // batch_size
vector_dim = train.shape[2]

train = train.reshape(-1, seq_len, vector_dim, 1)
test = test.reshape(-1, seq_len, vector_dim, 1)

'''
計算グラフ構築
'''
with tf.variable_scope("Input"):
  x = tf.placeholder(dtype=tf.float32, shape=[None, train.shape[1:]])
with tf.variable_scope("Target"):
  t = tf.placeholder(dtype=tf.float32, shape=[None, 2])
with tf.variable_scope("dropout"):
  keep_prob = tf.placeholder(dtype=tf.float32)
with tf.variable_scope("batch_normalization"):
  b_n_on = tf.placeholder(dtype=tf.bool)

h = x

with tf.variable_scope("conv-FIR"):
  h = tf.layers.conv2d(h, filters=16, kernel_size=(6, 1), padding='same',
                       activation=tf.nn.relu, use_bias=False)
  h = tf.nn.dropout(h, keep_prob=keep_prob)

  h = tf.layers.batch_normalization(h, training=b_n_on, axis=3)
  h = tf.layers.conv2d(h, filters=32, kernel_size=(6, 1), padding='same',
                       activation=tf.nn.relu, use_bias=False)
  h = tf.layers.max_pooling2d(h, pool_size=(2, 1), strides=(2, 1))
  h = tf.nn.dropout(h, keep_prob=keep_prob)

  h = tf.layers.batch_normalization(h, training=b_n_on, axis=3)
  h = tf.layers.conv2d(h, filters=freq, kernel_size=(4, 1), padding='same',
                       activation=tf.nn.relu, use_bias=False)
  h = tf.layers.max_pooling2d(h, pool_size=(2, 1), strides=(2, 1))
  h = tf.nn.dropout(h, keep_prob=keep_prob)


with tf.variable_scope("spatial_filter"):
  h = tf.layers.batch_normalization(h, training=b_n_on, axis=2)
  h = tf.layers.conv2d(h, filters=freq, kernel_size=(1, 5), padding='same',
                       activation=tf.nn.relu, use_bias=False)
  h = tf.layers.average_pooling2d(h, pool_size=(1, 5), strides=(1, 5))

# with tf.variable_scope("RNN"):
#   h = tf.unstack(h, None, 1)
#   cell = tf.contrib.rnn.LayerNormBasicLSTMCell(50, dropout_keep_prob=keep_prob)
#   att_cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=4)
#   h, _ = tf.contrib.rnn.static_rnn(att_cell, h, dtype=tf.float32)
#   h = h[-1]

with tf.variable_scope("dense"):
  h = tf.contrib.layers.flatten(h)
#  h = tf.layers.batch_normalization(h, training=b_n_on)
  h = tf.layers.dense(inputs=h, units=256, activation=tf.nn.relu)
#  h = tf.layers.batch_normalization(h, training=b_n_on)
  h = tf.layers.dense(inputs=h, units=2, activation=tf.nn.relu)
  y = tf.nn.softmax(h)

with tf.variable_scope("loss"):
  loss = tf.reduce_mean(-tf.reduce_sum(
                        t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                        reduction_indices=[1]))
  tf.summary.scalar("loss", loss)

with tf.variable_scope("train_step"):
  optimizer = tf.train.AdamOptimizer()
  train_step = optimizer.minimize(loss)

'''
学習
'''

print("-------Session initialize--------")
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
merged = tf.summary.merge_all()
sess.run(init)
tr_feed = {x: train, t: train_label, keep_prob: 1.0, b_n_on: False}

print("-------start training-------")
for epoch in range(epochs):
  X_, Y_ = shuffle(train, train_label)

  for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size

    sess.run(train_step, feed_dict={
        x: X_[start:end],
        t: Y_[start:end],
        keep_prob: k_prob,
        b_n_on: True
    })

  training_loss = sess.run(loss, feed_dict={
        x: train,
        t: train_label,
        keep_prob: 1.0,
        b_n_on: False
  })

  summary = sess.run(merged, feed_dict=tr_feed)
  writer.add_summary(summary, epoch)

  print("epoch:{}\n training_loss:{}".format(
        epoch, training_loss))

print("-------finish training-------")

hist = np.zeros(train_label.shape)
for i in range(num_sample):
  probability = sess.run(y, feed_dict={
        x: train,
        keep_prob: k_prob,
        b_n_on: False
  })
  inf = np.eye(2)[np.argmax(probability, 1)]
  hist += inf
hist /= num_sample
correct_prediction = np.equal(np.argmax(hist, 1), np.argmax(train_label, 1))
accuracy = np.mean(correct_prediction.astype(np.float32))

print("train_accuracy:{}".format(accuracy))

plt.plot(np.argmin(hist, 1))
plt.plot(np.argmin(train_label, 1))
plt.show()
