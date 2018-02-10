import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.pardir)
from data.date_ERD.load_erd import Load_erd
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


seq_len = 32

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


epochs = 100
batch_size = 1024
f_dim = train.shape[2]
lstm1_dim = 20
lstm2_dim = 20

model = Sequential()
model.add(Bidirectional(LSTM(lstm1_dim,
                             dropout = 0.2,
                             return_sequences = True), input_shape=(seq_len, f_dim)))
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
hist = model.fit(train, train_label,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_split=2/7,
                 callbacks=[csv_logger],
                 shuffle=True,
                 verbose=1)


score = model.evaluate(test, test_label, verbose=1)
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
