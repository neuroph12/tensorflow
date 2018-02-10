import numpy as np
from datautil import normal_data, split_data, elec_map2d_full, data_loader
import matplotlib.pyplot as plt
import seaborn
from model import FBConvLSTM2D, FBSpatialLSTM
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam


def get_data_loader(sub_range=[1, 85],
                   bpfilter=[0.5, 45],
                   problem='lr',
                   ):

    X, y = normal_data(sub_range=sub_range,
                       bpfilter=bpfilter,
                       problem=problem)

    X = elec_map2d_full(X)
    X = X.transpose(0,1,3,4,2)
    y = to_categorical(y)
    return  X, y

def main():
    train_sub_range = [1, 50]
    test_sub_range = [51, 60]
    bpfilter = None
    problem = '4'
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-5
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

    # model = FBConvLSTM2D(input_shape, classies)
    model = FBSpatialLSTM(input_shape, classies)
    model.model.compile(optimizer=Adam(lr=lr),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    model.model.fit(x=X_train, y=y_train,
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