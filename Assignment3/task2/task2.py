import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

import keras
import cv2
import numpy as np


X=np.load('dataU.npy')
Y=np.load('maskU.npy')
#
X=tf.image.resize_images(X, [256,256])
Y=tf.image.resize_images(Y, [256,256])

x_train=X[:9000]
x_test=X[9000:]
y_train=Y[:9000]
y_test=Y[9000:]
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255
#
#y_train = y_train.astype('float32') / 255
#y_test = y_test.astype('float32') / 255

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
del X
del Y


def image_generator(x_train,y_train, batch_size = 32):
    count= -batch_size
    y_train=np.expand_dims(y_train, axis=3)
    while True:
        # Select files (paths/indices) for the batch
        count+=batch_size
        if(count>=9000-batch_size):
            count=0
        batch_input = []
        batch_output = [] 

        # Read in each input, perform preprocessing and get labels
        for i in range(batch_size):
            input1 = x_train[i+count]
            output = y_train[i+count]

            # input = preprocess_input(image=input)
            batch_input += [ input1 ]
            batch_output += [ output ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )

        yield( batch_x, batch_y )



def mode(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


model = mode()
#model.summary()
#bs=1




yy=np.expand_dims(y_train, axis=3)
#trainGen=image_generator(x_train,y_train, batch_size = bs)
model.load_weights('modelt2.h5')
#H = model.fit_generator(trainGen,steps_per_epoch=int(9000/bs) ,epochs=1)


yyy=np.expand_dims(y_test, axis=3)

score = model.evaluate(x_test,yyy )
print(score)


#model.save('modelt2.h5')

#x_test=cv2.imread()
#y_pred = model.predict(x_test)
print(y_pred)

for i in range(100):
    cv2.imwrite('output3/'+str(i)+'_pred.png',y_pred[i]*255)
    cv2.imwrite('output3/'+str(i)+'_inp.png',x_test[i]*255)
    cv2.imwrite('output3/'+str(i)+'_gt.png',y_test[i]*255)


