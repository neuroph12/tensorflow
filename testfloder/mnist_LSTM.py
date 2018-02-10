from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.wrappers import Bidirectional
from numpy import *
from keras.preprocessing import sequence
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.regularizers import l1, l2
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def plot_history(history):
    
    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


model = Sequential()
model.add(BatchNormalization(input_shape = (28, 28)))
model.add(Bidirectional(LSTM(10, 
                             W_regularizer = l2(0.01),
                             dropout = 0.2,
                             return_sequences = True),
                        input_shape = (28, 28)))                      
model.add(Flatten())
model.add(BatchNormalization())                        
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.summary()
              
history = model.fit(X_train, Y_train, nb_epoch=100, batch_size=512, validation_data=(X_test, Y_test))

plot_history(history)
