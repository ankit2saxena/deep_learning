# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:32:47 2018

@author: Ankit
"""

import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import pandas as pd
import tensorflow as tf
#import matplotlib.pyplot as plt

import ImportDataUtil as DataHandler
import ResNet34Model as ResNet34

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint


if __name__ == '__main__':
    
    learning_rate = 5e-3
    decay = 5e-4
    momentum = 0.9
    batch_size = 32
    n_train = 4000
    n_val = 400
    
    model = ResNet34.ResNet34Model(learning_rate, momentum, decay)
    
    print(model.summary())
    
    training_data, training_videos, training_labels = DataHandler.get_training_data(n_train)
    print("Training Data loaded. Number of images: ", len(training_data))
    
    validation_data, validation_videos, validation_labels = DataHandler.get_validation_data(n_val)
    print("Validation Data loaded. Number of images: ", len(validation_data))

    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    
    validation_data = np.array(validation_data)
    validation_labels = np.array(validation_labels)
   
    print("Training Model...")
    if not os.path.exists("Model/"):
        os.mkdir("Model/")
        
    model_filepath="Model/ResNet34.bestweights.hdf5"
    model_checkpoint = ModelCheckpoint(model_filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    model_callbacks_list = [model_checkpoint]
    history = model.fit(training_data, training_labels, batch_size = batch_size, 
                        epochs = 200, validation_data = (validation_data, validation_labels),
                        callbacks = model_callbacks_list)
    print("Training completed successfully")
    
    columns = ['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error', 'mean_squared_error', 'val_mean_squared_error',
                'mean_absolute_percentage_error', 'val_mean_absolute_percentage_error', 'average_l1', 'val_average_l1']
    
    loss_dataframe = pd.DataFrame(columns = columns)
    loss_dataframe['loss'] = history.history['loss']
    loss_dataframe['val_loss'] = history.history['val_loss']
    loss_dataframe['mean_absolute_error'] = history.history['mean_absolute_error']
    loss_dataframe['val_mean_absolute_error'] = history.history['val_mean_absolute_error']
    loss_dataframe['mean_squared_error'] = history.history['mean_squared_error']
    loss_dataframe['val_mean_squared_error'] = history.history['val_mean_squared_error']
    loss_dataframe['mean_absolute_percentage_error'] = history.history['mean_absolute_percentage_error']
    loss_dataframe['val_mean_absolute_percentage_error'] = history.history['val_mean_absolute_percentage_error']
    loss_dataframe['average_l1'] = history.history['average_l1']
    loss_dataframe['val_average_l1'] = history.history['val_average_l1']
    
    loss_dataframe.to_csv("Model/resnet34_loss.csv", index=False)
    
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Model/resnet34_loss.jpeg')
    plt.show()
    
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Model/resnet34_loss_mae.jpeg')
    plt.show()
    
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Mean Squared Error')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Model/resnet34_loss_mse.jpeg')
    plt.show()
    
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('Mean Absoulte Percentage Error')
    plt.ylabel('MAPE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Model/resnet34_loss_mape.jpeg')
    plt.show()
    
    plt.plot(history.history['average_l1'])
    plt.plot(history.history['val_average_l1'])
    plt.title('Average L1 loss')
    plt.ylabel('average L1 loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('Model/resnet34_loss_average_l1.jpeg')
    plt.show()
    '''
    
    print("Saving model...")
    model.save('Model/resnet34.h5')
    print("Model Saved at location ../Model/resnet34.h5")
    
    #plot_model(model, to_file = "Model/resnet34.png", show_shapes = True)
