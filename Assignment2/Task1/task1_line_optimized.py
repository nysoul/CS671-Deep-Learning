
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
train_images=train_images.astype('float32')
#print(len(set(train_labels)))
train_labels=to_categorical(train_labels,dtype = 'int')
#	For Sequential Layers Stacking
from keras.models import Sequential

#	creating the model object
model = Sequential()

from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout


model.add(Conv2D(32, kernel_size=3, strides=1, activation='relu', input_shape=(28,28,3)))
model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.5))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

model.add(Flatten())

model.add(Dense(2056, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Dense(1024,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(96,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

history=model.fit(train_images, train_labels, batch_size=50, epochs= 1, validation_split=0.4)

##Learning Curves
import matplotlib.pyplot as plt
print("\n")
print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title("loss")
plt.legend()
plt.figure()

epochs = range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label = 'Training accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation accuracy')
plt.title("acurracy")
plt.legend()
plt.figure()

##Confusion Matrix and F1Score
from sklearn.metrics import confusion_matrix, f1_score
y_pred= model.predict(train_images)
y_pred=y_pred.argmax(axis=1)
cm = confusion_matrix(train_labels.argmax(axis=1),y_pred)
np.savetxt('cm1.txt',cm)
print(cm,'\n')
f1 = f1_score(train_labels.argmax(axis=1),y_pred, average = 'weighted')
print(f1,'\n')

