#!/usr/bin/env python
# coding: utf-8



import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import keras



train_images = np.load("imgs.npy")
train_label = np.load("labels.npy")




import sys

from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import scipy
from sklearn.model_selection import  train_test_split
import numpy.random as rng
#import argparse
#
#
#parser = argparse.ArgumentParser(description='Description of your program')
#parser.add_argument('--phase', help='Description for foo argument', required=True)
#args = vars(parser.parse_args())
#phase=args['phase']


def Regressor(input_img):
    reg_conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
    reg_conv1_1 = BatchNormalization()(reg_conv1_1)
    reg_conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2")(reg_conv1_1)
    reg_conv1_2 = BatchNormalization()(reg_conv1_2)
    reg_pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(reg_conv1_2)

    reg_conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(reg_pool1)
    reg_conv2_1 = BatchNormalization()(reg_conv2_1)
    reg_conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2")(reg_conv2_1)
    reg_conv2_2 = BatchNormalization()(reg_conv2_2)
    reg_pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(reg_conv2_2)

    reg_conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(reg_pool2)
    reg_conv3_1 = BatchNormalization()(reg_conv3_1)
    reg_conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(reg_conv3_1)
    reg_conv3_2 = BatchNormalization()(reg_conv3_2)
    reg_pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(reg_conv3_2)
        
    reg_flat = Flatten()(reg_pool3)
    fc1 = Dense(256, activation='relu')(reg_flat)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(16, activation='relu')(fc2)
    fc4 = Dense(2, activation='relu')(fc3)
    regress = Model(inputs = input_img, outputs =  fc4)
    return regress

x=train_images
y=train_label
x = x.reshape((4000, 560, 352, -1))

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.01)
regressorInput = Input((560, 352, 1))
regressor = Regressor(regressorInput)
regressor.compile(optimizer = Adam(0.0005), loss= 'mean_squared_error')
regressor.summary()

batch_size = 10
epochs = 2

regressor.load_weights('modelt3.h5')
regressor.predict(X_test)
#regressor.fit(X_train,Y_train, batch_size=batch_size,validation_split=0.2 ,epochs=epochs,verbose=1)
#egressor.save_weights('modelt3.h5')






