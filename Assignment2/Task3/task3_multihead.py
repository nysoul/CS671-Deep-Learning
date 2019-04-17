import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
imginput =[]
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
            imginput.append(img)




load_images_from_folder("data/") 

imginput = np.asarray(imginput)

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
model.load_weights("weights.h5")
#history=model.fit(imginput,[y1,y2,y3,y4],epochs = 3, validation_split = 0.33, shuffle = True , batch_size=32)
#model.save_weights("weights.h5")
model.summary()

##iNTERMEDIATE LAYERS

from keras import models
layers_output=[layer.output for layer in model.layers[1:6]]
activation_model=models.Model(inputs=model.input,outputs=layers_output)
for i in range (1,6):
#    print('For Image :',i) 
    image1=np.expand_dims(imginput[i*113],axis=0)   
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
        

layer_name = layer_names[4]
size = 28
filter_images=[]
for i in range(128):
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
