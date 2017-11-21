# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_center]: ./images/center_2017_11_15_11_55_53_218.jpg "Image (Center)"
[recover_1]: ./images/right_2016_12_01_13_45_13_320.jpg "Recovery Image"
[recover_2]: ./images/right_2016_12_01_13_45_13_420.jpg "Recovery Image"
[recover_3]: ./images/right_2016_12_01_13_45_13_522.jpg "Recovery Image"
[recover_4]: ./images/right_2016_12_01_13_45_13_623.jpg "Recovery Image"
[recover_5]: ./images/right_2016_12_01_13_45_13_724.jpg "Recovery Image"
[recover_6]: ./images/right_2016_12_01_13_45_13_825.jpg "Recovery Image"
[original]: ./images/original.jpg "Normal Image"
[flipped]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64 (model.py lines 59-74) 

The model includes RELU layers to introduce nonlinearity (lines 59-74), and the data is normalized in the model using a Keras lambda layer (code line 60). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 65). 

I performed many laps in order to reduce overfitting. I even tried many edge cases (recovering from off-track). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, and I started with learning rate 0.0001 (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used all the three cameras as guided in the videos. Also I tried as many laps as possible to make the model learn much from the data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Since most of the model codes are provided in the videos, I tried to generate high-quality data to make my model work well.

My model architecture is basically same as the model provided by Nvidia. I only added one Dropout layer to avoid overfitting problem.

In my first attempt the model worked poorly. I tried figure out by inspecting the code and found there was a problem in `generator` code. So I fixed it and confirmed the model then worked well. Still there were some curves the model had hard time overcoming them so I generated additional training data.

Now my model works well as the video (`run1.mp4`) shows. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-74) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Normalized image   					| 
| Cropping         		| (70, 25), (0, 0)          					| 
| Convolution 5x5     	| 24 filters, 1x1 stride, valid padding        	|
| RELU					|												|
| Convolution 5x5     	| 36 filters, 1x1 stride, valid padding        	|
| RELU					|												|
| Convolution 5x5     	| 48 filters, 1x1 stride, valid padding        	|
| RELU					|												|
| Dropout   	      	| 0.5 				                            |
| Convolution 3x3     	| 64 filters, 1x1 stride, valid padding        	|
| RELU					|												|
| Convolution 3x3     	| 64 filters, 1x1 stride, valid padding        	|
| RELU					|												|
| Fully connected		| ( (160-70-25)x320x3 )x100         								|
| Fully connected		| 100x50    									|
| Fully connected		| 50x10      									|
| Fully connected		| 10x1      									|

#### 3. Creation of the Training Set & Training Process

I used 80,594 images in total for training.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recognize somethings going wrong in such cases. These images show what a recovery looks like (recovering from righ-side bias:

![alt text][recover_1]
![alt text][recover_2]
![alt text][recover_3]
![alt text][recover_4]
![alt text][recover_5]
![alt text][recover_6]

To augment the data set, I also flipped images and angles thinking that this would help my model work also well to the right curve, since the original data is mostly composed of left cornering. For example, here is an image that has then been flipped:

![alt text][original]
![alt text][flipped]

Using all the three cameras and also augmented with the flipped images, I had the confidence that the amount of data wouldn't be a problem. So I trained a model, generate more **recovering** data for special cases, and now I succeed in making my virtual car pass the circuit. 