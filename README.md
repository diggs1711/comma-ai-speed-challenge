# comma-ai-speed-challenge

My attempt at the [comma ai speed challenge](https://github.com/commaai/speedchallenge). 
Using it as a learning tool as I improve my undestanding of image processing and deep learning.

## Introduction and Analysis
The aim of the challenege is to predict the speed of the car from the given video and speed readings. 

We are given two videos, one training video and one test video. Speed readings are only given for the training video with
the test serving as the evaluation video. The training video consists of 20400 frames and the test video of 10798 frames.
Both shot at 20FPS.

The train video is 17 minutes long whereas the the test is only 9 minutes. The train video starts off on a highway and
makes a turn off into a more residental area, slowing down as a consequence. This is reflected in the readings which show a
mean speed of 18.667 for the first 10200 frames of the video, whereas the second half of the video shows a mean speed of 
just 5.69. A good cross validation strategy will be required that incoroporates a validation set in both stages of the video. 
This will hopefully enable the model to generalise well to all conditions and speeds.

The test video consists of similar road conditions and terrain. Starting off slow in a residental area, moving to a highway 
and then back to residental. Not much different from the training video as far as I can tell.

## Feature Extraction
I'm currently running this on a standard colab gpu so memory is a concern. As a consequence of this, I have cut the video
size way down, from 480X640 to 100X440. This allows me to capture the majority of cars and road features,
while at the same time getting rid of unnecessary things such as the sky and the dashboard. I will come back to this 
and review it later on when I've got a decent setup and model working but for now it is a good comprimise.

## Model
I have setup a basic CNN with keras. Using the first 80% of the training video as the training set and the last 20% as the
validation set. The model consists of two Conv2D layers with 32 filters each and kernel sizes of 3x3 and 5x5 respectively.
A stride of 2 is used on both with relu as the activation function.
This is then flattened and fed into two Dense layers that output a single value.
Getting an mse of around 25-35 with 100 epochs. Obviously pretty shit and will definitely improve as I go on.

I will be researching more into combining a CNN model with an RNN model or possibly an LSTM. This will allow the model 
to capture the temporal nature of the videos and hopefully give a decent score.


## Update 11/05/20
Added Xception pretrained model as the first layer in the model. Using a TimeDistributed Layer from keras each time slice is fed into the model and then into a LSTM layer. This LSTM layer outputs a value at each time slice. This LSTM layer feeds into FC Layers and outputs a speed value.

Adding Xception helped tremendously bringing the MSE down below 20 on the validation set. 

I used OpenCV to place the predicted value from the test set onto the test video. This helped me to view and analyse the predicated test values and see where the main errors were. The first thing I noticed was that the values were fluctuating wildly between each frame. I wanted my model to realise that if the previous predicted value was 20, the next frame is probably not 8 but somewhere around 20 as well. To aid with this, I added the TimeDistributed layer and the LSTM layer. This helped bringing the MSE down to around 8/9 after 30 epochs. The values on the test video also seemed to improve and were noticeably more stable than the previous video.

Another error I noticed was at stop signs, the test video was still predicting values greater than 0, usually around 4-6. I presume this is due to the cropped video image not able to view the stop sign or road markings when stopped at the sign. Need to check the test video again and see if there's any stop signs

To deal with memory constraints on Colab I "chunked" the video into frames of 100 timesteps. This allowed me to batch frames instead of sending the whole video in as 1 timestep.



## Things to try
- Image augemetation
- k-fold validation with RNN model
- Check which frames have the biggest errors and figure out why

# Papers
https://arxiv.org/pdf/1604.07316v1.pdf

Will be updating this as I make progress...
