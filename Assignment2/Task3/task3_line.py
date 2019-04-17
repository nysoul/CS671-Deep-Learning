
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
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

#history=model.fit(train_images, train_labels, batch_size=50, epochs= 1, validation_split=0.4)
#model.save_weights('weights.line')
model.load_weights('weights.line')
##iNTERMEDIATE LAYERS

from keras import models
layers_output=[layer.output for layer in model.layers[1:6]]
activation_model=models.Model(inputs=model.input,outputs=layers_output)
for i in range (1,6):
#    print('For Image :',i) 
    image1=np.expand_dims(train_images[i*113],axis=0)   
    activations=activation_model.predict(image1)
    
    
    layer_names = []
    for layer in model.layers[1:6]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
      
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
       
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

#Convnet Filters
def deprocess_image(x):
    x-=x.mean()
    x/=(x.std()+1e-5)
    x*=0.1
    
    x+=0.5
    x=np.clip(x,0,1)
    
    x*=255
    x=np.clip(x,0,255).astype('uint8')
    return x
from keras import backend as K
def generate_pattern(layer_name, filter_index, size=28):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
#    cv2.imwrite("input_img.png",input_img_data)
    step = 1.
    for i in range(200):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        img = input_img_data[0]
        
    return deprocess_image(img)
    
    
layer_names = []
for layer in model.layers[1:6]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
        

layer_name = layer_names[1]
size = 28
filter_images=[]
for i in range(64):
        filter_img = generate_pattern(layer_name,i, size=size)
        filter_images.append(filter_img)
#        plt.imshow(filter_img)
        
images_per_row = 16
n_features = len(filter_images) # Number of filters
size = filter_images[0].shape[0] # Width or height of image
n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
display_grid = np.zeros((size * n_cols+n_cols*5, images_per_row * size+images_per_row*5,3))
for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
        channel_image = filter_images[col * images_per_row + row]
#         plt.imshow(channel_image)
        display_grid[col * size+col*5 : (col + 1) * size+(col)*5,row * size+row*5 : (row + 1) * size+row*5,:] = channel_image
cv2.imwrite('2_'+layer_names[0]+'.jpg',display_grid)
