import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.pardir)
from data.date_ERD.load_erd import Load_erd
from scipy.io import loadmat
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation, \
                                Convolution1D, Convolution2D, MaxPooling2D, Flatten, Dropout
from tensorflow.contrib.keras.python.keras.layers.wrappers import Bidirectional
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.regularizers import l1, l2
from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard


'''
データの生成
'''

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
label = np.array(label).astype(np.int32)
label = label.flatten()
label = np.eye(2)[label].astype(np.float32)

train_label = label
test_label = label

plot_x = np.r_[train, test]
plot_t = np.r_[train_label, test_label]

N_train = len(train)
vector_dim = train.shape[2]


epochs = 100
batch_size = 256
f_dim = train.shape[2]


model = Sequential()

model.add(Convolution2D(32, (2, 2), 2, padding='same', input_shape=train.shape[1:],
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.3))
model.add(Convolution2D(64, (2, 2), 2, padding='same',
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.3))
model.add(Activation("relu"))

model.add(Convolution2D(128, (2, 2), 2, padding='same',
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.3))
model.add(Convolution2D(256, (2, 2), 2, padding='same',
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(BatchNormalization(axis=3))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Convolution2D(512, (2, 2), 2, padding='same',
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Convolution2D(1024, (2, 2), 2, padding='same',
          kernel_regularizer = l2(0.005), use_bias=False))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(512, kernel_regularizer = l2(0.05)))
model.add(Activation("relu"))
model.add(Dropout(0.3))
model.add(Dense(128, kernel_regularizer = l2(0.05)))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.6))
model.add(Dense(2))

model.add(Activation("softmax"))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()



csv_logger = CSVLogger('training.log')
es_cb = EarlyStopping(monitor='val_loss', mode='auto', patience=30, verbose=1)
tb_cb = TensorBoard(log_dir='./logs', write_graph=True)
hist = model.fit(train, train_label,
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_split=1/7,
                 shuffle=True,
                 verbose=1,
                 callbacks=[es_cb, tb_cb, csv_logger])


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
plt.plot(time, y[:,1], label = 'prediction')
plt.plot(time, test_label[:,1], label = 'target')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('probability')
plt.show()
