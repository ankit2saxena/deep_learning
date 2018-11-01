# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:12:44 2018

@author: Ankit
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import BatchNormalization, add
from keras.optimizers import SGD
import keras.backend as K

normalization_axis = 3
batch_size = 32
eps = 1.1e-5

def average_l2(y_true, y_pred):
    return (1 - (1.0 / float(5.0 * batch_size)) * (K.sum(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1)))))

def average_l1(y_true, y_pred):
    return (1 - (1.0 / float(5.0 * batch_size)) * (K.sum(K.abs(y_true - y_pred))))


def convolution_block(input_tensor, kernel_size, conv_filters, strides = (2, 2)):
    conv_filter1, conv_filter2, conv_filter3 = conv_filters
    
    X = Conv2D(filters = conv_filter1, kernel_size = (1, 1), strides = strides, use_bias = False)(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = conv_filter2, kernel_size = kernel_size, padding = 'same', use_bias = False)(X)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = conv_filter3, kernel_size = (1, 1), padding = 'same', use_bias = False)(X)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    
    shortcut = Conv2D(filters = conv_filter3, kernel_size = (1, 1), strides = strides)(input_tensor)
    shortcut = BatchNormalization(epsilon = eps, axis = normalization_axis)(shortcut)
    
    X = add([X, shortcut])
    X = Activation('relu')(X)
    return X


def shortcut_block(input_tensor, kernel_size, conv_filters):
    conv_filter1, conv_filter2, conv_filter3 = conv_filters
    
    X = Conv2D(filters = conv_filter1, kernel_size = (1, 1), use_bias = False)(input_tensor)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = conv_filter2, kernel_size = kernel_size, padding = 'same', use_bias = False)(X)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = conv_filter3, kernel_size = (1, 1), padding = 'same', use_bias = False)(X)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis)(X)
    
    X = add([X, input_tensor])
    X = Activation('relu')(X)
    return X
    

def ResNet34Model(lr, m, d):
    inputImg = Input(shape = (224, 224, 3), name = "inputImg")
    
    #conv1
    X = ZeroPadding2D(padding = (3, 3))(inputImg)
    X = Conv2D(filters = 64, kernel_size = (7, 7), strides = (2, 2), use_bias = False, name = 'conv1')(X)
    X = BatchNormalization(epsilon = eps, axis = normalization_axis, name = 'conv1_bn')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'pool1')(X)
    
    X = convolution_block(X, 3, [64, 64, 256], strides = (1, 1))
    X = shortcut_block(X, 3, [64, 64, 256])
    X = shortcut_block(X, 3, [64, 64, 256])
    
    X = convolution_block(X, 3, [128, 128, 512])
    for i in range(1, 8):
        X = shortcut_block(X, 3, [128, 128, 512])
    
    X_full = AveragePooling2D((28, 28), name = 'global_avg_pool')(X)
    X_full = Flatten()(X_full)
    
    X_full = Dense(128, activation = 'relu', name = 'fc128')(X_full)
    X_full = Dense(5, activation = 'sigmoid', name = 'fc5')(X_full)
    
    
    model = Model(inputImg, X_full)
    sgd = SGD(lr = lr, decay = d, momentum = m, nesterov = True)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape', average_l1])
    
    return model
    pass
