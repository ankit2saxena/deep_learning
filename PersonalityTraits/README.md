
### Prediction of personality traits based on facial features extracted from videos using Deep Learning

“According to psychology researchers, the first impressions are formed in limited exposure (100ms) to unfamiliar faces.” [1]. But can an AI predict apparent personality traits of an individual given a short video?

During interviews, the mindset or perception of the interviewer can affect the selection of an individual. It is possible for machine learning models to classify the personality traits of an individual based on facial expressions, body language, or speech of the individual in the video. One of the most commonly used personality model is the Big-Five model which rates the five traits of Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism (OCEAN) [2].

The model generated here is trained and evaluated on YouTube videos provided by ChaLearn LAP APA2016 dataset [3]. 
It achieves excellent performance and is able to outperform the top teams from the competition on two categories: Openness, and Conscientiousness.

<hr />

#### [1] Prediction of Personality First Impressions with Deep Bimodal LSTM by Karen Yang, Stanford and Noa Glaser, Stanford, 2017.

#### [2] Big Five personality traits. In Wikipedia. Retrieved April 27, 2018.
https://en.wikipedia.org/wiki/Big_Five_personality_traits

#### [3] J.-I. Biel and D. Gatica-Perez. The YouTube lens: Crowdsourced personality impressions and audiovisual analysis of vlogs. Multimedia, IEEE Transactions on, 15(1):41– 55, 2013. 
http://chalearnlap.cvc.uab.es/dataset/20/description/#


<hr />

<pre>
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
</pre>

RUN "PlotLoss.py" to generate the loss graphs for 'total_loss', 'average_l1', 'loss_mae', 'loss_mape', 'loss_mse'.
