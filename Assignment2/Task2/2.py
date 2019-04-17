import cv2
import os
import numpy as np
input =[]
y1=[]
y2=[]
y3=[]
y4=[]
names=[]
i=0
def getlabel(filename):
    count = 0 
    label = ""
    names.append(filename)
    for c in filename:
        if c == '_':
            count+=1       
        else:
            if (count == 0):
                y1.append(int(c))
            if (count == 1):
                y2.append(int(c))
            if (count == 2):
                label = label+c
            if (count == 3):
                y4.append(int(c))  
    y3.append(label)

def load_images_from_folder(folder):
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        getlabel(filename)
        if img is not None:
            input.append(img)




load_images_from_folder("data/") 

input = np.asarray(input)

y1=np.array(y1)
y2=np.array(y2)
y3=np.array(y3)
y4=np.array(y4)

from keras.utils import to_categorical
y3=to_categorical(y3)


from keras.layers import Dense, Conv2D, Flatten,Input,MaxPooling2D,Dropout
from keras.models import Model

input_layer =Input(shape =(28,28,3),name='input')

x=Conv2D(32, kernel_size=3, activation='relu')(input_layer)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64, kernel_size=3, activation='relu')(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(128, kernel_size=3, activation='relu')(x)
x=MaxPooling2D((2,2))(x)
x=Dropout(0.6)(x)
x=Flatten()(x)

a1 = Dense(64, activation='relu')(x)
a1 = Dense(8, activation='relu')(a1)
a1 = Dense(1, activation='sigmoid',name='output1')(a1)


a2 = Dense(64, activation='relu')(x)
a2 = Dense(8, activation='relu')(a2)
a2 = Dense(1, activation='sigmoid',name='output2')(a2)

a3 = Dense(64, activation='relu')(x)
a3 = Dense(12, activation='softmax',name='output4')(a3)

a4 = Dense(64, activation='relu')(x)
a4 = Dense(8, activation='relu')(a4)
a4 = Dense(1, activation='sigmoid',name='output3')(a4)

model = Model(inputs=[input_layer], outputs=[a1,a2,a3,a4])
model.compile(optimizer='rmsprop',loss=['binary_crossentropy','binary_crossentropy','categorical_crossentropy','binary_crossentropy'], metrics=['accuracy'])

# And trained it via:
#model.load_weights("weights.h5")
history=model.fit(input,[y1,y2,y3,y4],epochs = 6, validation_split = 0.33, shuffle = True , batch_size=32)
#model.save_weights("weights.h5")
model.summary()

#Learning Curves
import matplotlib.pyplot as plt
print("\n")
print(history.history.keys())

output1_acc = history.history['output1_acc']
val_output1_acc = history.history['val_output1_acc']
output2_acc = history.history['output2_acc']
val_output2_acc = history.history['val_output2_acc']
output3_acc = history.history['output3_acc']
val_output3_acc = history.history['val_output3_acc']
output4_acc = history.history['output4_acc']
val_output4_acc = history.history['val_output4_acc']

epochs = range(1,len(output1_acc)+1)
plt.plot(epochs,output1_acc,'bo',label = 'Training output1 accuracy')
plt.plot(epochs,val_output1_acc,'b',label = 'Validation output1 accuracy')
plt.title("accuracy")
plt.legend()
plt.figure()

epochs = range(1,len(output2_acc)+1)
plt.plot(epochs,output2_acc,'bo',label = 'Training output2 accuracy')
plt.plot(epochs,val_output2_acc,'b',label = 'Validation output2 accuracy')
plt.title("accuracy")
plt.legend()
plt.figure()

epochs = range(1,len(output3_acc)+1)
plt.plot(epochs,output3_acc,'bo',label = 'Training output3 accuracy')
plt.plot(epochs,val_output3_acc,'b',label = 'Validation output3 accuracy')
plt.title("accuracy")
plt.legend()
plt.figure()

plt.plot(epochs,output4_acc,'bo',label = 'Training output4 accuracy')
plt.plot(epochs,val_output4_acc,'b',label = 'Validation output4 accuracy')
plt.title("accuracy")
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
output1_loss = history.history['output1_loss']
val_output1_loss = history.history['val_output1_loss']
output2_loss = history.history['output2_acc']
val_output2_loss = history.history['val_output2_loss']
output3_loss = history.history['output3_loss']
val_output3_loss = history.history['val_output3_loss']
output4_loss = history.history['output4_acc']
val_output4_loss = history.history['val_output4_loss']


epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs,val_loss,'b',label = 'Validation loss')
plt.title("loss")
plt.legend()
plt.figure()

epochs = range(1,len(output1_acc)+1)
plt.plot(epochs,output1_loss,'bo',label = 'Training output1 loss')
plt.plot(epochs,val_output1_loss,'b',label = 'Validation output1 loss')
plt.title("loss")
plt.legend()
plt.figure()


epochs = range(1,len(output2_acc)+1)
plt.plot(epochs,output2_loss,'bo',label = 'Training output2 loss')
plt.plot(epochs,val_output2_loss,'b',label = 'Validation output2 loss')
plt.title("loss")
plt.legend()
plt.figure()

epochs = range(1,len(output3_acc)+1)
plt.plot(epochs,output3_loss,'bo',label = 'Training output3 loss')
plt.plot(epochs,val_output3_loss,'b',label = 'Validation output3 loss')
plt.title("loss")
plt.legend()
plt.figure()

epochs = range(1,len(output4_acc)+1)
plt.plot(epochs,output4_loss,'bo',label = 'Training output4 loss')
plt.plot(epochs,val_output4_loss,'b',label = 'Validation output4 loss')
plt.title("loss")
plt.legend()
plt.figure()

##FSCORE
import tensorflow as tf
##Confusion Matrix
## Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
predictions = model.predict(input)
predictions[2] = np.argmax(predictions[2], axis = 1)
angle = np.argmax(y3, axis = 1)

for i in range(4):
    predictions[i]=predictions[i].astype('int')
    predictions[i]=predictions[i].reshape(predictions[i].shape[0])
cm1 = confusion_matrix(y1,predictions[0])
cm2 = confusion_matrix(y2,predictions[1])
cm3 = confusion_matrix(angle,predictions[2])
cm4 = confusion_matrix(y4,predictions[3])
print(cm1,'\n',cm2,'\n',cm3,'\n',cm4)
f1 = f1_score(y1,predictions[0], average = 'weighted')
f2 = f1_score(y2,predictions[1], average = 'weighted')
f3 = f1_score(angle,predictions[2], average = 'weighted')
f4 = f1_score(y4,predictions[3], average = 'weighted')
print(f1,'\n',f2,'\n',f3,'\n',f4)
#
#print("\n")
#print('Test loss:', round(score[0],6))
#print('Test accuracy:', score[1])