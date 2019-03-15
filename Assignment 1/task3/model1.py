import network
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255


test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

net=network.Network(784,10)
net.addlayer(784,100)
net.addlayer(100,30)
net.addlayer(30,10)
net.build()

training_data=zip(train_images,train_labels)
test_data=zip(test_images,test_labels)
net.fit(training_data,128,6,test_data=test_data)

