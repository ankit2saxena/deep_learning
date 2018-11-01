# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:27:28 2018

@author: Ankit
"""

import pandas as pd
import matplotlib.pyplot as plt


loss_dataframe = pd.read_csv('Model/resnet34_loss.csv')

loss = loss_dataframe['loss']
val_loss = loss_dataframe['val_loss']
mean_absolute_error = loss_dataframe['mean_absolute_error']
val_mean_absolute_error = loss_dataframe['val_mean_absolute_error']
mean_squared_error = loss_dataframe['mean_squared_error']
val_mean_squared_error = loss_dataframe['val_mean_squared_error']
mean_absolute_percentage_error = loss_dataframe['mean_absolute_percentage_error']
val_mean_absolute_percentage_error = loss_dataframe['val_mean_absolute_percentage_error']
average_l1 = loss_dataframe['average_l1']
val_average_l1 = loss_dataframe['val_average_l1']


plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss vs epoch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Model/resnet34_loss.jpeg')
plt.show()

plt.plot(mean_absolute_error)
plt.plot(val_mean_absolute_error)
plt.title('Mean Absolute Error vs epoch')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Model/resnet34_loss_mae.jpeg')
plt.show()

plt.plot(mean_squared_error)
plt.plot(val_mean_squared_error)
plt.title('Mean Squared Error vs epoch')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Model/resnet34_loss_mse.jpeg')
plt.show()

plt.plot(mean_absolute_percentage_error)
plt.plot(val_mean_absolute_percentage_error)
plt.title('Mean Absoulte Percentage Error vs epoch')
plt.ylabel('MAPE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Model/resnet34_loss_mape.jpeg')
plt.show()

plt.plot(average_l1)
plt.plot(val_average_l1)
plt.title('Average L1 loss vs epoch')
plt.ylabel('average L1 loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('Model/resnet34_loss_average_l1.jpeg')
plt.show()