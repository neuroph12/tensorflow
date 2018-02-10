import numpy as np
import matplotlib.pyplot as plt
import os, sys
from load_foot import Load_data
from scipy.io import loadmat
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, \
                                Convolution1D, Convolution2D, MaxPooling1D, Flatten, Dropout
from tensorflow.contrib.keras.python.keras.layers.wrappers import Bidirectional
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.regularizers import l1, l2
from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, CSVLogger

os.chdir('data')

seq_len = 128
def make_data(train_mat = "train_foot2.mat",
              test_mat = "test_foot2.mat",
              train_label_mat = "label_foot.mat",
              test_label_mat = "label_foot.mat"):
    data = Load_data(train_mat = train_mat,
                     test_mat = test_mat,
                     train_label_mat = train_label_mat,
                     test_label_mat = test_label_mat)
    train, train_label, test, test_label = data.get_data2d(seq_len)
    return train, train_label, test, test_label

train1, train_label1, test1, test_label1 = make_data(train_mat = "train_foot.mat",
                                                    test_mat = "test_foot.mat",
                                                    train_label_mat = "label_foot.mat",
                                                    test_label_mat = "label_foot.mat")
print(train1.shape)
print(test1.shape)

train = np.r_[train1, test1[:-4096]]
test = test1[-4096:]
train_label = np.r_[train_label1, test_label1[:-4096]]
test_label = test_label1[-4096:]

# train2, train_label2, test2, test_label2 = make_data(train_mat = "train_foot2.mat",
#                                                 test_mat = "test_foot2.mat",
#                                                 train_label_mat = "label_foot.mat",
#                                                 test_label_mat = "label_foot.mat")
#
# train = np.r_[train1, train2]
# train_label = np.r_[train_label1, train_label2]
# test = np.r_[test1, test2]
# test_label = np.r_[test_label1, test_label2]

#train = np.r_[train, test]
#train_label = np.r_[train_label, test_label]


epochs = 100
batch_size = 256
f_dim = train.shape[2]
lstm1_dim = 20
lstm2_dim = 20

model = Sequential()
model.add(Convolution1D(8, 4, input_shape = (seq_len, f_dim)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(3))
model.add(Convolution1D(12, 4, kernel_regularizer=l2(0.05)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(MaxPooling1D(3))
model.add(Convolution1D(18, 4, kernel_regularizer=l2(0.04)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(MaxPooling1D(3))
model.add(Bidirectional(LSTM(lstm1_dim,
                             dropout = 0.2,
                             return_sequences = True)))
model.add(Bidirectional(LSTM(lstm2_dim,
                             dropout = 0.2)))
model.add(BatchNormalization(axis = 1))
model.add(Dense(30, kernel_regularizer=l2(0.02)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(2, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.compile(optimizer = 'Nadam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()

csv_logger = CSVLogger('training.log')
print('num of training_data {}'.format(train.shape[0]))
print('num of test_data {}'.format(test.shape[0]))
hist = model.fit(train, train_label,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=(test, test_label),
                 callbacks = [csv_logger])


score = model.evaluate(test, test_label, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])

# plot results
plt.subplot(3, 1, 1)
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='loss')
plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(3, 1, 2)
loss = hist.history['acc']
val_loss = hist.history['val_acc']

epochs = len(loss)
plt.plot(range(epochs), loss, marker='.', label='acc')
plt.plot(range(epochs), val_loss, marker='.', label='val_acc')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')

y = model.predict(test)
time = range(len(y))
time = np.array(time).astype(np.float32)
plt.subplot(3, 1, 3)
plt.plot(time, y[:,0], label = 'prediction')
plt.plot(time, test_label[:,0], label = 'target')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('probability')
plt.show()
