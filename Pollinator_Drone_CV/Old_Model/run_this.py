# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import cv2                 # working with, mainly resizing, images
      # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.reset_default_graph()
#%%
LR = 1e-4
IMG_SIZE = 256


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 30, 5, activation='relu',name="conv1")
convnet = max_pool_2d(convnet, 3,name="pool1")

convnet = conv_2d(convnet, 64, 5, activation='relu',name="conv2")
convnet = max_pool_2d(convnet, 3,name="pool2")
#
convnet = conv_2d(convnet, 128, 5, activation='relu',name="conv3")
convnet = max_pool_2d(convnet, 5,name="pool3")
#
convnet = conv_2d(convnet, 64, 5, activation='relu',name="conv4")
convnet = avg_pool_2d(convnet, 5,name="pool4")
#
#convnet = conv_2d(convnet, 32, 5, activation='relu',name="conv5")
#convnet = avg_pool_2d(convnet, 5,name="pool5")

convnet = fully_connected(convnet, 256, activation='relu',name="fc1")
#convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax',name="fc2")


convnet = regression(convnet, optimizer='sgd', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=3)


#%%
model.load('my_model.tflearn')


#%%

path = '/home/aman/Documents/PersonalProjects/Pollinator_Drone_CV'

img = cv2.imread('check2.jpg',0)
img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

test=img.reshape(-1,IMG_SIZE,IMG_SIZE,1)
#%%
a=model.predict({'input': test})
if_closed_flower=np.argmax(a)

print("[DETECTION RESULT]: {:}".format(if_closed_flower))
