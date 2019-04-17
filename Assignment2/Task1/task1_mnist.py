
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)

from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#	For Sequential Layers Stacking
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D


model.add(Conv2D(32, kernel_size=7, strides=1, activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

model.add(Flatten())

model.add(Dense(1024, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
model.add(Dense(128,activation='softmax'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=50, epochs= 6, validation_split=0.4)
model.save_weights('weights.t1')
#model.load_weights('weights.t1')
##Plotting Learning Curves

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

##Confusion Matrix
from sklearn.metrics import confusion_matrix, f1_score
y_pred= model.predict(x_train)
y_pred=y_pred.argmax(axis=1)
cm = confusion_matrix(y_train.argmax(axis=1),y_pred)

print(cm,'\n')
f1 = f1_score(y_train.argmax(axis=1),y_pred, average = 'weighted')
print(f1,'\n')
