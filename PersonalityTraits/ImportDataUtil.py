# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:16:25 2018

@author: Ankit
"""

import os
import numpy as np
import pandas as pd
import imageio

training_video_path = "Data/Videos/Training/"
training_input_labels_path = "Data/training_gt.csv"
training_image_path = "Data/Images/Training/"
training_output_labels_path = "Data/training_final.csv"

test_video_path = "Data/Videos/Test/"
tset_input_labels_path = "Data/validation_gt.csv"
test_image_path = "Data/Images/Test/"
test_output_labels_path = "Data/test_final.csv"

validation_video_path = "Data/Videos/Validation/"
validation_input_labels_path = "Data/validation_gt.csv"
validation_image_path = "Data/Images/Validation/"
validation_output_labels_path = "Data/validation_final.csv"


training_images_filenames = []
validation_images_filenames = []
test_images_filenames = []

if os.path.exists(training_image_path):
    training_images_filenames = os.listdir(training_image_path)
    
if os.path.exists(validation_image_path):
    validation_images_filenames = os.listdir(validation_image_path)
    
if os.path.exists(test_image_path):
    test_images_filenames = os.listdir(test_image_path)
    
def scaleImage(image):
    
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    
    #image = image / 255.0
    image = np.divide(image, 255.0)
    
    for i in range(0, 3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / float(sd[i])
    
    return image
    

def get_training_data(n_samples):
    training_images = []
    training_names = []
    training_labels = []

    temp_dataframe = pd.read_csv(training_output_labels_path)
    
    for row_num in range(0, temp_dataframe.shape[0]):
        
        if len(training_names) >= n_samples:
            break
        
        row = temp_dataframe.loc[row_num, :]
        videoName, E, A, C, N, O, imageName = row.values
        
        if not os.path.exists(training_image_path+imageName):
            continue
        
        train_image = imageio.imread(training_image_path+imageName)
        train_image = np.array(train_image)
        train_image = scaleImage(train_image)
        training_images.append(train_image)
        training_names.append(videoName)
        training_labels.append([O, C, E, A, N])
    
    return [training_images, training_names, training_labels]
    pass


def get_validation_data(n_samples):
    validation_images = []
    validation_names = []
    validation_labels = []

    temp_dataframe = pd.read_csv(validation_output_labels_path)
    
    for row_num in range(0, temp_dataframe.shape[0]):
        
        if len(validation_names) >= n_samples:
            break
        
        row = temp_dataframe.loc[row_num, :]
        videoName, E, A, C, N, O, imageName = row.values
        
        if not os.path.exists(validation_image_path+imageName):
            continue
        
        val_image = imageio.imread(validation_image_path+imageName)
        val_image = np.array(val_image)
        val_image = scaleImage(val_image)
        validation_images.append(val_image)
        validation_names.append(videoName)
        validation_labels.append([O, C, E, A, N])
    
    return [validation_images, validation_names, validation_labels]
    pass


def get_test_data(n_samples):
    test_images = []
    test_names = []
    test_labels = []

    temp_dataframe = pd.read_csv(test_output_labels_path)
    
    for row_num in range(0, temp_dataframe.shape[0]):
        
        if len(test_names) >= n_samples:
            break
        
        row = temp_dataframe.loc[row_num, :]
        videoName, E, A, C, N, O, imageName = row.values
        
        if not os.path.exists(test_image_path+imageName):
            continue
        
        test_image = imageio.imread(test_image_path+imageName)
        test_image = np.array(test_image)
        test_image = scaleImage(test_image)
        test_images.append(test_image)
        test_names.append(videoName)
        test_labels.append([O, C, E, A, N])
    
    return [test_images, test_names, test_labels]
    pass
