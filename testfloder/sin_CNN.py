import pandas as pd
import numpy as np
import math
import random

random.seed(0)
# 乱数の係数
random_factor = 0.05
# サイクルあたりのステップ数
steps_per_cycle = 80
# 生成するサイクル数
number_of_cycles = 50

df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle)+ random.uniform(-1.0, +1.0) * random_factor))
df[["sin_t"]].head(steps_per_cycle * 2).plot()

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev = 100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)
    
from keras.models import Sequential  
from keras.layers import Dense, Activation, Conv1D, Flatten 
import matplotlib.pyplot as plt

in_out_neurons = 1
hidden_neurons = 1
seq_len = 100

(X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev =seq_len)  


model = Sequential()  
model.add(Conv1D(hidden_neurons, kernel_size = seq_len, input_shape = (seq_len, in_out_neurons)))
model.add(Flatten())
#model.add(Dense(in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
model.fit(X_train, y_train, batch_size=100, nb_epoch=100, validation_split=0.05) 

print(X_test.shape)
predicted = model.predict(X_test) 
dataf =  pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
dataf.plot(figsize=(15, 5))
plt.show()
