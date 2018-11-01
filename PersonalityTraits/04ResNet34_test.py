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

from keras.models import load_model
from keras.optimizers import SGD

def predictLabels(model, test_data, test_actual, testVideoName):
    
    O, C, E, A, N = 0.0, 0.0, 0.0, 0.0, 0.0
    
    for i in range(0, num_samples):
        test_data_sample = test_data[i]
        test_data_sample = np.expand_dims(test_data_sample, axis = 0)
        test_predicted = model.predict(test_data_sample)[0]
        O += float(test_predicted[0])
        C += float(test_predicted[1])
        E += float(test_predicted[2])
        A += float(test_predicted[3])
        N += float(test_predicted[4])
        
    O = O / float(num_samples)
    C = C / float(num_samples)
    E = E / float(num_samples)
    A = A / float(num_samples)
    N = N / float(num_samples)
    
    test_predicted = np.array([O, C, E, A, N])
    
    print("Video Details: ", testVideoName)
    columns = ['Openness', 'Conscientiousness', 'Extroversion', 'Agreeableness', 'Neuroticism']
    indexes = ['Actual', 'Predicted']
    
    test_details = np.vstack([test_actual, test_predicted])
    test_details = pd.DataFrame(data = test_details, index = indexes, columns = columns)
    print(test_details)
    return test_predicted


def calculate_loss(actual, predicted):
    columns = ['O', 'C', 'E', 'A', 'N']
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    diff = np.abs(actual - predicted)
    traits_mean = np.mean(diff, axis = 1)
    
    print("Accuracies: ")
    for i in range(0, 5):
        print("Trait", columns[i], ":", (1 - float(traits_mean[i])))
    
    print("Total : ", 1 - float(np.mean(traits_mean)))


if __name__ == '__main__':
    
    learning_rate = 5e-3
    decay = 5e-4
    momentum = 0.9
    
    num_samples = 5
    n_test = 250
    
    print("Loading model...")
    #model = load_model('Model/resnet34.h5')
    model_filepath="Model/ResNet34.bestweights.hdf5"
    model = ResNet34.ResNet34Model(learning_rate, momentum, decay)
    model.load_weights(model_filepath)
    sgd = SGD(lr = learning_rate, decay = decay, momentum = momentum, nesterov = True)
    model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mse', 'mae', 'mape', ResNet34.average_l1])
    print("Model loaded successfully")
    
    
    test_data, test_videos, test_labels = DataHandler.get_test_data(n_test)
    print("Test Data loaded. Number of images: ", len(test_data))

    #test_data = test_data[0: 10]
    #test_videos = test_videos[0: 10]
    #test_labels = test_labels[0: 10]
    
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    total_predicted = []
    total_actual = []
    
    print("Starting Testing...")
    test_counter = 0
    while test_counter < len(test_labels):
        test_data_sample = test_data[test_counter: (test_counter + num_samples)]
        testVideoName = test_videos[test_counter]
        for i in range(test_counter, (test_counter + num_samples)):
            if testVideoName != test_videos[i]:
                print("FAILED...Video ID mismatch.")
                test_counter += num_samples
                continue
        
        test_labels_pred = predictLabels(model, test_data_sample, test_labels[test_counter], testVideoName)
        total_actual.append(test_labels[test_counter])
        total_predicted.append(test_labels_pred)
        test_counter += num_samples
        print("************************************************")

    calculate_loss(total_actual, total_predicted)