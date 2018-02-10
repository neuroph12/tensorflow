import os, sys
sys.path.append(os.pardir)
import numpy as np
import tensorflow as tf
from module.utility.history import History, Real_time_plot, EarlyStopping
from module.basics.my_layers import bidirectional_LSTM, time_stacked_conv1d
from module.basics.my_layers import accuracy, loss, training, stacked_affine
from sklearn.utils import shuffle
from data.date_ERD.load_erd import Load_erd


'''
モデルハイパーパラメータ
'''
seq_len = 32
lr = 1e-4
epochs = 100
batch_size = 256
# cell
cell = 'GRU' # GRU, LSTM, TF-LSTM, Grid-LSTM
# conv parameters
conv_kernels = [3,3,3,3,3,3,3,3]
conv_filters = [4,4,4,4,4,4,4,4]
conv_dropout = 0.2
pool_size = 2
# rnn parameters
lstm_n_hidden = 4
n_attention = 0
# dense parameters
affine_kernels = [10, 2]
aff_dropout = 0.2
# early_stopping
stop = False


'''
データの生成
'''
os.chdir('data/date_ERD')

train_data = Load_erd(data_name='data_151116_zr01.mat')
train, train_label = train_data.get_data2d(seq_len=seq_len)

test_data = Load_erd(data_name='data_151116_zr02.mat')
test, test_label = test_data.get_data2d(seq_len=seq_len)

plot_x = np.r_[train, test]
plot_t = np.r_[train_label, test_label]

N_train = len(train)
vector_dim = train.shape[2]


"""
define model
"""

with tf.variable_scope("Input"):
  x = tf.placeholder(dtype = tf.float32, shape = [None, seq_len, vector_dim])
with tf.variable_scope("Target"):
  t = tf.placeholder(dtype = tf.float32, shape = [None, 2])


h = x

with tf.variable_scope("conv1D_layers"):
  conv_keep_prob = tf.placeholder(dtype = tf.float32)
  h = time_stacked_conv1d(h,
                          filters=conv_filters,
                          kernels=conv_kernels, activation=tf.nn.relu,
                          pool_size=pool_size, keep_prob=conv_keep_prob)

h = tf.unstack(h, None, 1)

# with tf.variable_scope("bidirectional_LSTM1"):
#   h = bidirectional_LSTM(h, n_hidden=lstm_n_hidden, cell = cell,
#                          return_seq=True, attention=n_attention)
with tf.variable_scope("bidirectional_LSTM2"):
  h = bidirectional_LSTM(h, n_hidden=lstm_n_hidden, cell = cell,
                         return_seq=False, attention=n_attention)

with tf.variable_scope("classification_layers"):
  aff_keep_prob = tf.placeholder(dtype = tf.float32)
  h = stacked_affine(h, affine_kernels=affine_kernels, keep_prob=aff_keep_prob)
  y = tf.nn.softmax(h)

with tf.variable_scope("Evaluation_layers"):
  loss = loss(t = t, y = y)
  accuracy = accuracy(t = t, y = y)
  tf.summary.scalar("loss", loss)
  tf.summary.scalar("accuracy", accuracy)

train_step = training(loss = loss, lr = lr)

"""
learning
"""

if __name__ == '__main__':
  '''
  モデル学習
  '''

  init = tf.global_variables_initializer()
  config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  sess = tf.InteractiveSession(config = config)
  sess.run(init)

  writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph)
  merged = tf.summary.merge_all()

  n_batches = N_train // batch_size
  history = History()
  ax = Real_time_plot()
  early_stopping = EarlyStopping(patience=30, verbose=1)

  tr_feed = {x: train, t:train_label, aff_keep_prob:1.0, conv_keep_prob:1.0}
  val_feed = {x: test, t:test_label, aff_keep_prob:1.0, conv_keep_prob:1.0}

  for epoch in range(epochs):
    X_, Y_ = shuffle(train, train_label)

    if epoch % 10 == 0:
      # Plot data and functions
      digit = sess.run(y, feed_dict = {
                                       x: plot_x,
                                       t: plot_t,
                                       aff_keep_prob: 1.0,
                                       conv_keep_prob: 1.0})


      ax.plot(np.argmax(digit, axis=1))

    for i in range(n_batches):
      start = i * batch_size
      end = start + batch_size
      sess.run(train_step, feed_dict={
                                      x: X_[start:end],
                                      t: Y_[start:end],
                                      aff_keep_prob: 1 - aff_dropout,
                                      conv_keep_prob: 1 - conv_dropout
                                      })

    tr_loss = loss.eval(session=sess, feed_dict=tr_feed)
    tr_acc = accuracy.eval(session=sess, feed_dict=tr_feed)
    val_loss = loss.eval(session=sess, feed_dict=val_feed)
    val_acc = accuracy.eval(session=sess, feed_dict=val_feed)

    history(epoch = epoch,
            val_acc = val_acc, val_loss = val_loss,
            tr_acc = tr_acc, tr_loss = tr_loss)

    if stop:
      if early_stopping.validate(val_loss):
        break

    summary = sess.run(merged, feed_dict=tr_feed)
    writer.add_summary(summary, epoch)

  history.plot()
