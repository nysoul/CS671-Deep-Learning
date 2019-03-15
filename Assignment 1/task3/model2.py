import network


import cv2
import os
import numpy as np
def getlabel(filename):
    name = ""
    count = 0 
    id = 0 
    for c in filename:
        if c == '_':
            count += 1
            if (count == 1):
                id += int(name)*1
            if (count == 2):
                id += int(name)*2
            if (count == 3):
                id += int(name)*4
            if (count == 4):
                id += int(name)*48
                break
            name  = ""
        else:
            name = name+c
        
        
    return id

def load_images_from_folder(folder):
    images = []
    labels=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        labels.append(int(getlabel(filename)))
        if img is not None:
            images.append(img)
    return (images,labels)
train_images,train_labels=load_images_from_folder("data/")



train_images=np.array(train_images)

from keras.utils import to_categorical
train_images=train_images.reshape((96000,28*28*3))
train_images=train_images.astype('float32')/255
print(len(set(train_labels)))
train_labels=to_categorical(train_labels,dtype = 'int')

net=network.Network(2352,96)
net.addlayer(2352,1000)
net.addlayer(1000,300)
net.addlayer(300,96)
net.build()

training_data=zip(train_images,train_labels)
test_data=zip(train_images,train_labels)
net.fit(training_data,250,10,test_data=test_data)
