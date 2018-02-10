import numpy as np
from datautil import normal_data, split_data, elec_map2d_full, data_loader, crop_data
import matplotlib.pyplot as plt
import seaborn
from model import CascadeConvRNN
from tensorflow.python.keras.utils import to_categorical

def get_data_loader(sub_range=[1, 85],
                   bpfilter=[0.5, 45],
                   problem='lr',
                   ):

    X, y = crop_data(sub_range=sub_range, bpfilter=bpfilter, problem=problem,
                     time_window=0.5, time_step=0.25)

    X = elec_map2d_full(X)
    X = X.transpose(0,1,3,4,2)
    y = to_categorical(y)
    print(X.shape)
    return  X, y

def main():
    train_sub_range = [1, 50]
    test_sub_range = [51, 60]
    bpfilter = None
    problem = '4'
    batch_size = 128 * 2
    lr = 1e-3
    epoch = 500
    name = None
    early_stop = 3



    X_train, y_train = get_data_loader(
        sub_range=train_sub_range,
        bpfilter=bpfilter,
        problem=problem,
        )

    input_shape = X_train.shape[1:]
    
    if problem == '4':
        classies = 4
    else:
        classies = 2

    model = CascadeConvRNN(input_shape, classies)
    model.model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    model.model.fit(x=X_train, y=y_train, batch_size=batch_size,
                    epochs=epoch, shuffle=True, validation_split=0.2)

    X_test, y_test = get_data_loader(
        sub_range=test_sub_range,
        bpfilter=bpfilter,
        problem=problem,
        )

    score = model.model.evaluate(X_test, y_test, verbose=0)

    print('test_accuracy:{}'.format(score[1]))


if __name__ == '__main__':
    main()