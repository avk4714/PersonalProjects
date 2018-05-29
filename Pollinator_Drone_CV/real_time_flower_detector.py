# CSE 576 Project
# [TITLE] : Pollinator Drone flower detection algorithm with CNN
# [AUTHORS] : Melanie Anderson, Yogesh Chukewad, Aman Kalia
# [DESCRIPTION] - The code applies real time detection and identification of
# the state of a flower in the camera frame. These states are "OPEN", "CLOSED"
# and "NOT DETECTED". These detections are feasible due to a CNN trained using
# images of the tulip flower in both closed and open states. The model was
# on the TensorFlow platform through Python. Inspiration for real time detection
# was drawn from Adrian Rosebrock's work on Real Time Object detection with deep
# neural network

# Import packages

# Packages for video stream IO
from imutils.video import VideoStream
from imutils.video import FPS

# Packages for Neural Net, Math operations, Computer vision
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Packages for dealing with directories
import os
from random import shuffle
from tqdm import tqdm

# Neural Network - CNN
tf.reset_default_graph()
LR = 1e-4
IMG_SIZE = 256
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#Pool 1
convnet = conv_2d(convnet, 30, 5, activation='relu',name="conv1")
convnet = max_pool_2d(convnet, 3,name="pool1")
#Pool 2
convnet = conv_2d(convnet, 64, 5, activation='relu',name="conv2")
convnet = max_pool_2d(convnet, 3,name="pool2")
#Pool 3
convnet = conv_2d(convnet, 128, 5, activation='relu',name="conv3")
convnet = max_pool_2d(convnet, 5,name="pool3")
#Pool 4
convnet = conv_2d(convnet, 64, 5, activation='relu',name="conv4")
convnet = avg_pool_2d(convnet, 5,name="pool4")
convnet = fully_connected(convnet, 256, activation='relu',name="fc1")
convnet = fully_connected(convnet, 2, activation='softmax',name="fc2")
convnet = regression(convnet, optimizer='sgd', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_verbose=3)

print("[INFO] Loading CNN model ...")
model.load('my_model.tflearn')


# Gather live video stream
    # Initialize VideoStream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Capturing video Frame
while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ##cropped = image[70:170, 440:540]
    ##cv2.imshow("cropped", cropped)
    tframe = cv2.resize(gframe, (IMG_SIZE,IMG_SIZE))
    rframe = tframe.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    pframe = model.predict({'input': rframe})
    if_closed_flower=np.argmax(pframe)

    # Frame print
    # output Message
    if if_closed_flower == 0 and np.any(pframe[0] >= 0.82) == True:
        msg = "OPEN"
    elif if_closed_flower == 0 and np.all(pframe[0] >= 0.82) == False:
        msg = "NOT DETECTED"
    elif if_closed_flower == 1:
        msg = "CLOSED"

    label = "{}:{}".format("Flower Status",msg)
    label2 = "{}:{}".format("Probability",pframe)
    cv2.putText(frame, label, (5,260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 128, 2)
    #cv2.putText(frame, label2, (5,290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 156, 2)
    #output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    #Update fps
    fps.update()

fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# Clean Up
cv2.destroyAllWindows()
vs.stop()
