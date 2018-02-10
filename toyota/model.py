import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Conv3D, Flatten, Dense
from tensorflow.python.keras.layers import BatchNormalization, ConvLSTM2D, Bidirectional, LSTM
from tensorflow.python.keras.layers import AveragePooling3D, Dropout, TimeDistributed
from tensorflow.python.keras.regularizers import l1, l2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

class FBSpatialLSTM(object):
    def __init__(self, input_shape, output_classes, filter_dims=75):
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.filter_dims = filter_dims
        self.model = self.build()

    def build(self):

        input_seq = Input(shape=self.input_shape)

        x = Conv3D(filters=32, kernel_size=(self.filter_dims, 1, 1), padding='same', use_bias=False)(input_seq)
        x = BatchNormalization(-1)(x)
        x = AveragePooling3D(pool_size=(self.filter_dims, 1, 1), strides=(int(self.filter_dims/3), 1, 1), padding='valid')(x)
        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)
        x = Bidirectional(ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=True))(x)
        x = Bidirectional(ConvLSTM2D(filters=32, kernel_size=(3, 3), return_sequences=False))(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        logits = Dense(self.output_classes, activation='softmax')(x)
       
        model = Model(inputs=input_seq, outputs=logits)
        model.summary()
        return model

class FBConvLSTM2D(object):

    def __init__(self, input_shape, output_classes, filter_dims=75):
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.filter_dims = filter_dims
        self.model = self.build()

    def build(self):

        input_seq = Input(shape=self.input_shape)

        x = Conv3D(filters=32, kernel_size=(self.filter_dims, 1, 1),
                padding='same', use_bias=False, kernel_regularizer=l1(1e-3))(input_seq)
        x = BatchNormalization(-1)(x)
        x = AveragePooling3D(pool_size=(self.filter_dims, 1, 1), strides=(int(self.filter_dims/3), 1, 1), padding='valid')(x)
        x = ConvLSTM2D(filters=64, kernel_size=(3, 3), return_sequences=False, padding='same')(x)
        
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)
        
        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)

        x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization(-1)(x)
        x = Dropout(0.5)(x)
        
        x = Conv2D(filters=1024, kernel_size=(5, 5), activation='relu', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization(-1)(x)        
        x = Dropout(0.5)(x)
        
        x = Flatten()(x)
        logits = Dense(self.output_classes, activation='softmax', kernel_regularizer=l2(1e-3))(x)

        model = Model(inputs=input_seq, outputs=logits)
        model.summary()
        
        return model



class CascadeConvRNN(object):
    def __init__(self, input_shape, output_classes):
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.model = self.build()

    def build(self):

        input_seq = Input(shape=self.input_shape)

        x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))(input_seq)
        x = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))(x)
        x = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))(x)
        
        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(1024))(x)
        x = TimeDistributed(Dropout(0.5))(x)
        
        x = LSTM(units=64, return_sequences=True)(x)
        x = LSTM(units=64, return_sequences=False)(x)
        
        x = Dense(1024)(x)
        x = Dropout(0.5)(x)

        logits = Dense(self.output_classes, activation='softmax')(x)

        model = Model(inputs=input_seq, outputs=logits)
        model.summary()
        
        return model
