Libraries Required:

numpy
pandas
imageio
matplotlib
os
keras
tensorflow
pypi
pydotplus
graphviz
pydot
cv2
face_recognition
moviepy
random

Keep the same directory structure for the code and data files.
RUN "02DataSplit.ipynb" to generate the images from the videos along with the csv file with the labels for each image.
RUN "03ResNet34_train.py" to train the ResNet34 model on the extracted images. The model weights are saved at location "Model/ResNet34.bestweights.hdf5". The model loss values for each epoch are stored in "resnet34_loss.csv".
RUN "PlotLoss.py" to generate the loss graphs for 'total_loss', 'average_l1', 'loss_mae', 'loss_mape', 'loss_mse'.
RUN "04ResNet34_test.py" to predict the average OCEAN scores for each image in the test dataset using the extracted images, with the ResNet34 model.

RUN "06LSTMResNet34_train.py" to train the LSTM model on the extracted images. The model weights are saved at location "Model/LSTMResNet34.bestweights.hdf5". The model loss values for each epoch are stored in "LSTMresnet34_loss.csv".
RUN "PlotLoss.py" to generate the loss graphs for 'total_loss', 'average_l1', 'loss_mae', 'loss_mape', 'loss_mse'.
RUN "07LSTMResNet34_test.py" to predict the average OCEAN scores for each image in the test dataset using the extracted images, with the LSTM model.