# comma-ai-speed-challenge

My attempt at the [comma ai speed challenge](https://github.com/commaai/speedchallenge). 
Using it as a learning tool as I improve my undestanding of image processing and deep learningz.

## Introduction and Analysis
The aim of the challenege is to predict the speed of the car from the given video and speed readings. 

We are given two videos, one training video and one test video. Speed readings are only given for the training video with
the test serving as the evaluation video. The training video consists of 20400 frames and the test video of 10798 frames.
Both shot of 20FPS.

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
while at the same time getting rid of unnecessary things such as the sky and dashboard. I will come back to this 
and review it later on when I've got a decent setup and model working but for now it is a good comprimise.

## Model
I setup a basic CNN with keras. Using first 80% of the training video as the training set and the last 20% as the
validation set. The model consists of two Conv2D layers with 32 filters each and kernel sizes of 3x3 and 5x5 respectively.
A stride of 2 is used on both with relu as the activation function.
This is then flattened and fed into two Dense layers that output a single value.
Getting an mse of around 25-35 with 100 epochs. Obviously pretty shit and will definitely improve as I go on.

I will be researching more into combining a CNN model with an RNN model or possibly an LSTM. This will allow the model 
to capture the temporal nature of the videos and hopefully give a decent score.

## Things to try
- Image augemetation
- k-fold validation with RNN model
- Timeseriessplit validation from sckit-learn could be useful

Will be updating this as I make progress...
