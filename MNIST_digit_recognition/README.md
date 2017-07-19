# MNIST_digit_recognition

### Comparison of various machine learning algorithms used to predict the digit in an image (MNIST dataset).

MNIST dataset consits of 55000 training images, 10000 test images, and 5000 images for validation.

The accuracy shown below has been calculated on the validation set. Also, in each IPython Notebook, the accuracy on the test set is shown for each epoch.

## Accuracy for various techniques:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/MNIST_digit_recognition/images/accuracy.png)

## Loss for various techniques:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/MNIST_digit_recognition/images/loss.png)

## Labels:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/MNIST_digit_recognition/images/labels.png)

Regularization techniques like Max norm and Dropout provide the maximum accuracy in the tests performed (The plots for accuracy and loss were visualized in TensorBoard).
### Convolutional Neural Network implementation in TensorFlow exceeds 99.00% accuracy on the MNIST dataset.

## To start Tensorboard, use command:
### tensorboard --logdir=/tmp/model

and open http://0.0.0.0:6006 in your browser.
