# notMNIST_image_recognition

### Comparison of various machine learning algorithms used to predict the character in an image (notMNIST dataset).

notMNIST dataset consits of 200000 training images, 10000 test images, and 10000 images for validation.

The accuracy shown below has been calculated on the validation set. Also, in each IPython Notebook, the accuracy on the test set is shown for each epoch.

## Accuracy for various techniques:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/notMNIST_image_recognition/images/accuracy.png)

## Loss for various techniques:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/notMNIST_image_recognition/images/loss.png)

## Labels:
![](https://github.com/ankit2saxena/tensorflow_examples/blob/master/notMNIST_image_recognition/images/labels.png)

Regularization techniques like Dropout and Learning Rate scheduling provide the maximum accuracy in the tests performed (The plots for accuracy and loss were visualized in TensorBoard).
### Convolutional Neural Network implementation with Dropout in TensorFlow have the maximum accuracy on notMNIST dataset.

## To start Tensorboard, use command:
### tensorboard --logdir=/tmp/notmnist

and open http://0.0.0.0:6006 in your browser.
