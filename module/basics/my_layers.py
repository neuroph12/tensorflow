import tensorflow as tf
from tensorflow.contrib import rnn


def bidirectional_LSTM(x, n_hidden, return_seq=False, attention=0,
                       cell = 'LSTM'):
    # maybe x.shape = (batch_size, seq_len, dim)
    # change x to list of (batch_size, dim)
    #x = tf.unstack(x, None, 1)

    if cell == 'GRU':
        cell_forward = rnn.GRUCell(n_hidden)
        cell_backward = rnn.GRUCell(n_hidden)

    if cell == 'LSTM':
        cell_forward = rnn.LSTMCell(n_hidden)
        cell_backward = rnn.LSTMCell(n_hidden)

    if cell == 'TF-LSTM':
        cell_forward = rnn.TimeFreqLSTMCell(num_units=n_hidden,
                                            feature_size=3,
                                            frequency_skip=1)
        cell_backward = rnn.TimeFreqLSTMCell(num_units=n_hidden,
                                             feature_size=3,
                                             frequency_skip=1)

    if cell == 'Grid-LSTM':
        cell_forward = rnn.GridLSTMCell(n_hidden,
                                        num_frequency_blocks = [5])
        cell_backward = rnn.GridLSTMCell(n_hidden,
                                         num_frequency_blocks = [5])


    if attention == 0:
        pass
    else:
        cell_forward = rnn.AttentionCellWrapper(cell_forward,
                                                attn_length=attention)
        cell_backward = rnn.AttentionCellWrapper(cell_backward,
                                                 attn_length=attention)

    h, _, _ = \
        rnn.static_bidirectional_rnn(cell_forward, cell_backward, x,
                                     dtype=tf.float32)
    if return_seq == True:
        return h
    else:
        return h[-1]


def time_stacked_conv1d(x,
                        filters, kernels, activation=tf.nn.relu,
                        pool_size=4, keep_prob=1.0):
    h = x
    idx = 0
    for filter_, kernel_ in zip(filters, kernels):
        idx += 1
        with tf.variable_scope("conv1d" + str(idx)):
            h = tf.layers.conv1d(h, filter_, kernel_, activation=activation)
            h = tf.nn.dropout(x=h, keep_prob=keep_prob)
    idx += 1
    with tf.variable_scope("conv1d" + str(idx)):
        h = tf.layers.max_pooling1d(h, pool_size=pool_size, strides=pool_size)
    return h


def stacked_affine(x, affine_kernels = [10, 2], stddev=1.0,
                   keep_prob=1.0, batch_norm=None, activation=tf.nn.relu):
    h = x

    for i in range(len(affine_kernels) - 1):
        with tf.variable_scope("dense" + str(i)):
            h = tf.layers.dense(h, affine_kernels[i],
                kernel_initializer = tf.truncated_normal_initializer(stddev=stddev))
            if batch_norm != None:
              h = tf.layers.batch_normalization(h, training=batch_norm)
            h = activation(h)
            h = tf.nn.dropout(x=h, keep_prob=keep_prob)
    with tf.variable_scope("dense_last"):
        h = tf.layers.dense(h, affine_kernels[-1],
            kernel_initializer = tf.truncated_normal_initializer(stddev=stddev))
    return h


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    with tf.name_scope('accracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def loss(t, y):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                        t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                        reduction_indices=[1]))
    return cross_entropy

def training(loss, lr = 1e-3):
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    train_step = optimizer.minimize(loss)
    return train_step

def bayes_predictor(logits, sample):
    def infer(logits):
        label = tf.argmax(logits, 1)
        one_hot_label = tf.one_hot(label, depth=logits.shape[1])
        return one_hot_label
    i = tf.constant(0)
    cond = lambda i : tf.less(i, sample)
    logits_list = tf.while_loop(cond, infer, [logits])
    predict = tf.reduce_mean(tf.stack(logits_list, axis=0), axis=0)
    return predict
