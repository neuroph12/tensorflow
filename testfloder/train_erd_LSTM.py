import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from sklearn.utils import shuffle
from loaddata import Load_data
import os
import shutil


LOG_DIR = os.path.join(os.path.dirname(__file__), 'log')
shutil.rmtree(LOG_DIR)
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)


np.random.seed(0)
tf.set_random_seed(1234)


def inference(x, n_in=None, n_time=None, n_hidden=None, n_out=None, keep_prob=None):
    def weight_variable(shape, name = 'W'):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name)

    def bias_variable(shape):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial)

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_in])
    x = tf.split(x, n_time, 0)

    with tf.name_scope('RNN'):
        cell_forward = rnn.GRUCell(n_hidden)
        cell_forward = rnn.AttentionCellWrapper(cell_forward, attn_length = 14)
        cell_backward = rnn.GRUCell(n_hidden)
        cell_backward = rnn.AttentionCellWrapper(cell_backward, attn_length = 14)

        h, _, _ = \
            rnn.static_bidirectional_rnn(cell_forward, cell_backward, x,
                                         dtype=tf.float32)
        h = h[-1]
        h = tf.nn.dropout(h, keep_prob)
    with tf.name_scope('fc_NN'):
        W = weight_variable([n_hidden * 2, n_hidden], name = 'W')
        b = bias_variable([n_hidden])
        h = tf.nn.elu(tf.layers.batch_normalization(tf.matmul(h, W) + b))
    
        h = tf.nn.dropout(h, keep_prob)
    
        Wo = weight_variable([n_hidden, n_out], name = 'Wo')
        bo = bias_variable([n_out])
        y = tf.nn.softmax(tf.layers.batch_normalization(tf.matmul(h, Wo) + bo))

    W_list = [W, Wo]
    return y, W_list


def loss(y, t, W_list, reg = [1e-3, 5e-4, 1e-3]):
    with tf.name_scope('loss'):
        cross_entropy = \
            tf.reduce_mean(-tf.reduce_sum(
                            t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                            reduction_indices=[1])) \
            + reg[0] * (tf.nn.l2_loss(W_list[0]) + tf.nn.l2_loss(W_list[1])) \
            + reg[1] * (tf.reduce_sum(tf.abs(W_list[0])) + tf.reduce_sum(tf.abs(W_list[1]))) \
            + reg[2] * (tf.reduce_sum(tf.reduce_max(tf.abs(W_list[0]), axis = 1)) + tf.reduce_sum(tf.reduce_max(tf.abs(W_list[1]), axis = 1)))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def loss2(y, t, W_list, reg = [1e-3, 5e-4, 1e-3]):
    with tf.name_scope('loss'):
        hinge_loss = \
            tf.losses.hinge_loss(labels = t, logits = y) \
            + reg[0] * (tf.nn.l2_loss(W_list[0]) + tf.nn.l2_loss(W_list[1])) \
            + reg[1] * (tf.reduce_sum(tf.abs(W_list[0])) + tf.reduce_sum(tf.abs(W_list[1]))) \
            + reg[2] * (tf.reduce_sum(tf.reduce_max(tf.abs(W_list[0]), axis = 1)) + tf.reduce_sum(tf.reduce_max(tf.abs(W_list[1]), axis = 1)))
    tf.summary.scalar('hinge_loss', hinge_loss)
    return hinge_loss


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.999)
    with tf.name_scope('optimizer'):
        train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    with tf.name_scope('accracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


if __name__ == '__main__':
    '''
    データの生成
    '''
    plot_data = Load_data()
    plot_x, plot_test, plot_t = plot_data.get_data2d()
    
    data = Load_data()
    train, test, label = data.get_data2d()
#    train, test, label = data.get_one_data2d()
    n = len(train)

    '''
    モデル設定
    '''
    n_in = 28
    n_time = 28
    n_hidden = 100
    n_out = 2
    reg = [5e-1, 5e-3, 4e-3]
    keep_node = 0.15    
    epochs = 500
    batch_size = 1024
    N_train = len(train)

    x = tf.placeholder(tf.float32, shape=[None, n_time, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])
    keep_prob = tf.placeholder(tf.float32)
    y, W = inference(x,
                  n_in=n_in,
                  n_time=n_time,
                  n_hidden=n_hidden,
                  n_out=n_out,
                  keep_prob = keep_prob)
    loss = loss2(y, t, W, reg)
    train_step = training(loss)

    accuracy = accuracy(y, t)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    history = {
        'val_loss': [],
        'val_acc': [],
        'tr_loss': [],
        'tr_acc': []
    }

    '''
    モデル学習
    '''

    init = tf.global_variables_initializer()
    sess = tf.Session()
    file_writer = tf.summary.FileWriter(LOG_DIR)
    file_writer.add_graph(sess.graph)
    summaries = tf.summary.merge_all()
    sess.run(init)

    n_batches = N_train // batch_size

    # Set up figure.
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show()
    plt.grid()

    for epoch in range(epochs):
        X_, Y_ = shuffle(train, label)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end],
                keep_prob: keep_node
            })
        
        if epoch % 10 == 0:
            # Plot data and functions
            digit = sess.run(y, feed_dict = {
                                             x: plot_x,
                                             t: plot_t,
                                             keep_prob: 1.0})
            plt.cla()
            ax.plot(digit)
            plt.grid()
            plt.draw()
            plt.pause(1.0 / 60.0)                    
        
        
        tr_loss = loss.eval(session=sess, feed_dict={
            x: train,
            t: label,
            keep_prob: 1.0
        })
        tr_acc = accuracy.eval(session=sess, feed_dict={
            x: train,
            t: label,
            keep_prob: 1.0
        })
        
        val_loss = loss.eval(session=sess, feed_dict={
            x: test,
            t: label,
            keep_prob: 1.0
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: test,
            t: label,
            keep_prob: 1.0
        })

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)        

        print('epoch:', epoch,
              ' tr_loss:', tr_loss,
              ' tr_acc:', tr_acc,
              ' val_loss:', val_loss,
              ' val_acc:', val_acc)

#        if early_stopping.validate(val_loss):
#            break

    '''
    学習の進み具合を可視化
    '''
    v_loss = history['val_loss']
    t_loss = history['tr_loss']
    v_acc = history['val_acc']
    t_acc = history['tr_acc']
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(v_loss)), v_loss,
             label='validation_loss', color='red')
    plt.plot(range(len(t_loss)), t_loss,
             label='training_loss', color='black')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy')
    
    fig = plt.figure()
    plt.plot(range(len(v_acc)), v_acc,
             label='validation_acc', color='red')
    plt.plot(range(len(t_acc)), t_acc,
             label='training_acc', color='black')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    '''
    予測精度の評価
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: test,
        t: label,
        keep_prob: 1.0
    })
    print('accuracy: ', accuracy_rate)
    
    prob = sess.run(y, feed_dict={
                x: test,
                keep_prob: 1.0})
    plt.plot(prob[:,1])
    plt.plot(label[:,1])
    plt.show()
