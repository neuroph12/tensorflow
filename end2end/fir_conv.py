import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.pardir)

from scipy.io import loadmat
from tensorflow.contrib.keras.python.keras.layers.recurrent import LSTM
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Activation, \
                         Conv2D, MaxPooling2D, Flatten, Dropout, Reshape
from tensorflow.contrib.keras.python.keras.layers.wrappers import Bidirectional
from tensorflow.contrib.keras.python.keras.preprocessing import sequence
from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from tensorflow.contrib.keras.python.keras.regularizers import l1, l2
from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping, CSVLogger
from load_foot import Load_data, make_data

'''
モデルハイパーパラメータ
'''
seq_len = 128
freq = 64
lr = 1e-4
epochs = 1000
batch_size = 512
k_prob = 0.8
num_sample = 100

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

N_train = len(train)
vector_dim = train.shape[2]
train = train.reshape(-1, seq_len, vector_dim, 1)
test = test.reshape(-1, seq_len, vector_dim, 1)


'''
construction model
'''
inputs = Input(shape=(train.shape[1:]))

x = Conv2D(16, (8, 1), padding='same', use_bias=False)(inputs)
x = Conv2D(16, (8, 1), padding='same', use_bias=False, activation='relu')(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(0.3)(x)

x = Conv2D(32, (8, 1), padding='same', use_bias=False)(inputs)
x = Conv2D(32, (8, 1), padding='same', use_bias=False, activation='relu')(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (8, 1), padding='same', use_bias=False)(inputs)
x = Conv2D(64, (8, 1), padding='same', use_bias=False, activation='relu')(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(0.3)(x)

x = Conv2D(64, (1,5), padding='valid', use_bias=False, activation='relu')(x)

x = Reshape((128, 64))(x)
x = Bidirectional(LSTM(20,
                       dropout = 0.2,
                       return_sequences = False))(x)

#x = Flatten()(x)

x = Dense(256, use_bias=False, activation='relu')(x)
x = BatchNormalization(axis=-1)(x)
x = Dropout(0.5)(x)
logits = Dense(2, use_bias=False)(x)

model = Model(inputs=inputs, outputs=logits)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(train, train_label, validation_split=2/7,
                    batch_size=batch_size, epochs=epochs, shuffle=True)
