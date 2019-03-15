"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""
import tensorflow as tf

import random

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
class Network(object):

    def __init__(self,inputsize,outputsize):
        self.X = tf.placeholder("float", [None,inputsize])
        self.Y = tf.placeholder("float", [None,outputsize])
        self.weights= {}
        self.biases = {}
        self.numberoflayers = 0
        self.weightnames=[]
        self.biasnames=[]
        self.train_step = 0
        self.correct_pred = 0
        self.accuracy = 0
        self.cross_entropy = 0
        self.predict = 0
    def addlayer(self,inputshape,outputshape):
        weightname = "w"+str(self.numberoflayers+1)
        biasname = "b"+str(self.numberoflayers+1)
        self.weightnames.append(weightname)
        self.biasnames.append(biasname)
        self.weights[weightname]=tf.Variable(tf.truncated_normal([inputshape, outputshape], stddev=0.1))
        self.biases[biasname]=tf.Variable(tf.constant(0.1, shape=[outputshape]))
        self.numberoflayers+=1
        
    def build(self):
        count = self.numberoflayers
        i = 0
        while(i<count):
            if i==0:
                layer = tf.add(tf.matmul(self.X, self.weights[self.weightnames[i]]), self.biases[self.biasnames[i]])
                i+=1
            else:
                layer = tf.add(tf.matmul(layer, self.weights[self.weightnames[i]]), self.biases[self.biasnames[i]])
                i+=1
            
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=layer))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_pred = tf.equal(tf.argmax(layer, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.predict = tf.argmax(layer, 1)
        
    def fit(self,training_data,mini_batch_size,epochs,test_data=None):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        train_images = []
        train_labels = []
        for images,labels in training_data:
            train_images.append(images)
            train_labels.append(labels)
        n = len(train_images)
        for i in range(epochs):
            mini_batch_images = [train_images[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            mini_batch_labels = [train_labels[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for batch_x,batch_y in zip(mini_batch_images,mini_batch_labels):
                sess.run(self.train_step, feed_dict={self.X: batch_x, self.Y: batch_y})
        

        # print loss and accuracy (per minibatch)

            minibatch_loss, minibatch_accuracy = sess.run([self.cross_entropy, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
            print("Epochs", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))
            
        if test_data:
            test_images = []
            test_labels = []
            for images,labels in test_data:
                test_images.append(images)
                test_labels.append(labels)
            test_accuracy = sess.run(self.accuracy, feed_dict={self.X: test_images, self.Y: test_labels})
            print("\nAccuracy on test set:", test_accuracy)
            pred = self.predict.eval({self.X: test_images} , session = sess)
            print(pred.shape)
            test_labels = [ np.argmax(t) for t in test_labels ]

            print("confusion matrix: \n",confusion_matrix(test_labels, pred))
#            np.savetxt('cmat.txt',confusion_matrix,fmt='%.2f')
            print("f1 score:",f1_score(test_labels,pred, average="weighted"))
            
            title = "Learning Curves (Naive Bayes)"
            # Cross validation with 100 iterations to get smoother mean test and train
            # score curves, each time with 20% data randomly selected as a validation set.
            cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
            
            estimator = GaussianNB()
            self.plot_learning_curve(estimator, title,test_images[:1000],test_labels[:1000], cv=cv)
            
            
    def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        return plt

            

 
        
        
        
        

     