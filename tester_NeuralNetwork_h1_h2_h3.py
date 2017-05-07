#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Università degli studi di Trento (Tn) - Italy
Center for Mind/Brain Sciences CIMeC
Language, Interaction and Computation Laboratory CLIC

@author: Patrizio Bellan
         patrizio.bellan@gmail.com
         patrizio.bellan@studenti.unitn.it

         github.com/patriziobellan86


        Francesco Mantegna
        fmantegna93@gmail.com
          
"""
from __future__ import division
from __future__ import print_function

import csv
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from Word2Vec import Word2Bigrams, Pos2Vec

w2v = Word2Bigrams()
p2v = Pos2Vec()

#==============================================================================
# Parameters
#==============================================================================
model_ext = ".ckpt"

trainDim = 0.7
testDim = 0.2
validDim = 0.1

# Training Parameters
learning_rate = 0.05 #ok # 0.001
training_epochs = 50 #ok # 80
batch_size = 50 #ok
display_step = 10

# Network Parameters
n_hidden_1 = 250#ok #1000#500 # 1st layer number of features
n_hidden_2 = 125#ok #250#500 # 2nd layer number of features
n_hidden_3 = 25#50#500 # 2nd layer number of features

n_input = len(w2v.bidict)
n_classes = len(p2v.posdict)

# optimazer params
beta1=0.9
beta2=0.999
epsilon=1e-02 #1e-08 ok


def CreateFigure(s, trainAcc,testAcc, loss, folder):
    # accuracy figure
    plt.figure()     
    plt.plot(trainAcc, 'b-', label='training accuracy',linewidth=1.0)
    plt.plot(testAcc, 'm-', label='test accuracy',linewidth=1.0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy value')
    plt.ylim(0.0, 1.1)
    plt.yticks(np.arange(0.0, 1.1, 0.05))
    
    plt.tick_params(axis='both', which='major', labelsize=8)
    s_accuracy = 'accuracy: '+str(testAcc[0])
    plt.annotate(s_accuracy, xy=(25, testAcc[0]), xytext=(10, (testAcc[0]-0.3)),
            arrowprops=dict(facecolor='magenta', shrink=0.05), ) 
    
    plt.title(s)
    # add legend
    plt.legend(loc='best')
    # Save Figure
    plt.savefig(folder+'accuracy_'+s+'.jpeg',dpi=150,format='jpeg', bbox_inches="tight")

    # loss figure
    plt.figure()
    plt.plot(loss, 'r-',linewidth=0.1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(s)
    # Save Figure
    plt.savefig(folder+'loss_'+s+'.jpeg',dpi=150,format='jpeg', bbox_inches="tight")
    # close all the figure to save memory
    plt.close('all')


def SaveResults (filename, data):
    with open(filename, "a") as outfile:
        writer = csv.writer(outfile, delimiter=';', quotechar='"')
        # write headers
        writer.writerow(data)
        writer.writerow('\n')   

#==============================================================================
#==============================================================================
# extended test
#==============================================================================
def GlobalTestingNNs():
    for learning_rate in [0.001, 0.05]:
        for training_epochs in [50, 100]:
            for batch_size in [50, 80]:
                for n_hidden_1 in [250, 500, 1000]:
                    for n_hidden_2 in [125, 250, 500]:
                        for n_hidden3 in [25, 50, 500]:
                            for beta1 in [0.5, 0.9]:
                                for beta2 in [0.09, 0.999]:
                                    for epsilon in [1e-02, 1e-08]:
                                        h1(f,learning_rate,training_epochs, 
                                           batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                                           n_input,n_classes,beta1,beta2,epsilon,
                                           '3layers_')
                                        h2(f,learning_rate,training_epochs, 
                                           batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                                           n_input,n_classes,beta1,beta2,epsilon,
                                           '2layers_')
                                        h3(f,learning_rate,training_epochs, 
                                           batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                                           n_input,n_classes,beta1,beta2,epsilon,
                                           '3layers_')
#==============================================================================
# short test
#==============================================================================
def TestingNNs(f):
    learning_rate = 0.05
    batch_size = 50
    beta1= 0.9
    beta2 = 0.999
    training_epochs = 50
    
    for epsilon in [1e-02, 1e-08]:  
        for n_hidden_1 in [250, 500]:
             h1(f,learning_rate,training_epochs, 
                   batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                   n_input,n_classes,beta1,beta2,epsilon,
                   '1layers_')
            for n_hidden_2 in [125, 250]:
                h2(f,learning_rate,training_epochs, 
                   batch_size,n_hidden_1, n_hidden_2,0,
                   n_input,n_classes,beta1,beta2,epsilon,
                   '2layers_')
                for n_hidden3 in [25, 50]:
                    h3(f,learning_rate,training_epochs, 
                       batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                       n_input,n_classes,beta1,beta2,epsilon,
                       '3layers_')

#==============================================================================
# H1
#==============================================================================
                                        
def h1(f, learning_rate,training_epochs, batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                n_input,n_classes,beta1,beta2,epsilon,
                filename):    
    
    # tf Graph input
    x = tf.placeholder("float", shape=(None, n_input))
    y = tf.placeholder("float", shape=(None, n_classes))
    
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)        
        # Output layer with linear activation
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        
        return out_layer
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)+1e-50) # avoid NaN
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1, beta2=beta2,epsilon=epsilon).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
       
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess = tf.Session()    
    sess.run(init)
    
    # Training cycle
    epochs_loss=[]     # store epochs_loss for plot
    epochs_training_Accuracy = []
    
    # avoid session inconsistency
    train.reset_epoch()
    test.reset_epoch()

    for epoch in range(training_epochs):
        epoch_loss = []
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            epoch_loss.append(avg_cost)
        # Display logs per epoch step
        if epoch % display_step == 0:
            s = 'Epoch: '+str(epoch+1)+' loss= '+str(avg_cost)
            if f:
                print(s, end="\n", file=f)     
            print (s)
            
        training_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y}, session=sess)
        epochs_training_Accuracy.append(training_accuracy)         
        if epoch % display_step == 0:
            s = 'accuracy epoch '+str(epoch+1)+': '+str(training_accuracy)    
            if f:
                print(s, end="\n", file=f)     
            print (s)
        
        f.flush()
        os.fsync(f.fileno())
        
        epochs_loss.append(epoch_loss)
        train.reset_epoch()
    totalLoss= avg_cost
#    print("Optimization Finished!")
    # Test model    
    batch_x, batch_y = test.next_batch(len(test.word))
    feed_dict={x: batch_x, y: batch_y}
    test_accuracy =  accuracy.eval(feed_dict=feed_dict, session=sess)         
    training_accuracy = sum(epochs_training_Accuracy)/len(epochs_training_Accuracy)
    
    # close session
    sess.close()
    
    s = 'training accuracy: '+str(training_accuracy)+' test accuracy: '+str(test_accuracy) + \
        ' difference (training-test): '+str(test_accuracy - training_accuracy) + \
        ' global loss: '+str(totalLoss)
    if f:
        print(s, end="\n", file=f)     
    print (s)
    f.flush()
    os.fsync(f.fileno())
        # num2str in scientific notation
    epsilon = '%.3e' % epsilon
    # composing stringName
    s = filename+'h1_'+str(n_hidden_1)+'h2_'+str(0)+'h3_'+str(0)+ \
        'learnRate_'+str(learning_rate)+'epochs_'+str(training_epochs)+ \
        'batchs_'+str(batch_size)+'b1_'+str(beta1)+'b2_'+str(beta2)+'ep_'+str(epsilon)
        
    line = [learning_rate,training_epochs, batch_size,n_hidden_1, 0,0,
            n_input,n_classes,beta1,beta2,epsilon,
            totalLoss,training_accuracy, test_accuracy,test_accuracy-training_accuracy]
    SaveResults(folder+filename+'.csv', line)
    
    test_accuracy = [test_accuracy for i in range(len(epochs_training_Accuracy))]
    CreateFigure(s, epochs_training_Accuracy,test_accuracy, epochs_loss, folder)

#==============================================================================
# H2
#==============================================================================
                                        
def h2(f, learning_rate,training_epochs, batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                n_input,n_classes,beta1,beta2,epsilon,
                filename):    
    
    # tf Graph input
    x = tf.placeholder("float", shape=(None, n_input))
    y = tf.placeholder("float", shape=(None, n_classes))
    
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        
        return out_layer
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)+1e-50) # avoid NaN
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1, beta2=beta2,epsilon=epsilon).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
       
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess = tf.Session()    
    sess.run(init)
    
    # Training cycle
    epochs_loss=[]     # store epochs_loss for plot
    epochs_training_Accuracy = []
    
    # avoid session inconsistency
    train.reset_epoch()
    test.reset_epoch()

    for epoch in range(training_epochs):
        epoch_loss = []
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            epoch_loss.append(avg_cost)
        # Display logs per epoch step
        if epoch % display_step == 0:
            s = 'Epoch: '+str(epoch+1)+' loss= '+str(avg_cost)
            if f:
                print(s, end="\n", file=f)     
            print (s)
            
        training_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y}, session=sess)
        epochs_training_Accuracy.append(training_accuracy)         
        if epoch % display_step == 0:
            s = 'accuracy epoch '+str(epoch+1)+': '+str(training_accuracy)    
            if f:
                print(s, end="\n", file=f)     
            print (s)
        
        f.flush()
        os.fsync(f.fileno())
        
        epochs_loss.append(epoch_loss)
        train.reset_epoch()
    totalLoss= avg_cost
#    print("Optimization Finished!")
    # Test model    
    batch_x, batch_y = test.next_batch(len(test.word))
    feed_dict={x: batch_x, y: batch_y}
    test_accuracy =  accuracy.eval(feed_dict=feed_dict, session=sess)         
    training_accuracy = sum(epochs_training_Accuracy)/len(epochs_training_Accuracy)
    
    # close session
    sess.close()
    
    s = 'training accuracy: '+str(training_accuracy)+' test accuracy: '+str(test_accuracy) + \
        ' difference (training-test): '+str(test_accuracy - training_accuracy) + \
        ' global loss: '+str(totalLoss)
    if f:
        print(s, end="\n", file=f)     
    print (s)
    f.flush()
    os.fsync(f.fileno())
        # num2str in scientific notation
    epsilon = '%.3e' % epsilon
    # composing stringName
    s = filename+'h1_'+str(n_hidden_1)+'h2_'+str(n_hidden_2)+'h3_'+str(n_hidden3)+ \
        'learnRate_'+str(learning_rate)+'epochs_'+str(training_epochs)+ \
        'batchs_'+str(batch_size)+'b1_'+str(beta1)+'b2_'+str(beta2)+'ep_'+str(epsilon)
        
    line = [learning_rate,training_epochs, batch_size,n_hidden_1, n_hidden_2,n_hidden_3,
            n_input,n_classes,beta1,beta2,epsilon,
            totalLoss,training_accuracy, test_accuracy,test_accuracy-training_accuracy]
    SaveResults(folder+filename+'.csv', line)
    
    test_accuracy = [test_accuracy for i in range(len(epochs_training_Accuracy))]
    CreateFigure(s, epochs_training_Accuracy,test_accuracy, epochs_loss, folder)
    
#==============================================================================
# H3
#==============================================================================

def h3 (f, learning_rate,training_epochs, batch_size,n_hidden_1, n_hidden_2,n_hidden3,
                n_input,n_classes,beta1,beta2,epsilon,
                filename):
    
    # tf Graph input
    x = tf.placeholder("float", shape=(None, n_input))
    y = tf.placeholder("float", shape=(None, n_classes))
    
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        
        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        layer_3 = tf.nn.relu(layer_3)
        
        # Output layer with linear activation
        out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
        
        return out_layer
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)+1e-50) # avoid NaN
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1, beta2=beta2,epsilon=epsilon).minimize(cost)
    
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer() 
    sess = tf.Session()    
    sess.run(init)
 
    # Training cycle
    epochs_loss=[]     # store epochs_loss for plot
    epochs_training_Accuracy = []
    # avoid session inconsistency
    train.reset_epoch()
    test.reset_epoch()
    
    for epoch in range(training_epochs):
        epoch_loss = []
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            epoch_loss.append(avg_cost)
        # Display logs per epoch step
        if epoch % display_step == 0:
            s = 'Epoch '+str(epoch+1)+' loss= '+str(avg_cost)
            if f:
                print(s, end="\n", file=f)     
            print (s)
            
        training_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y}, session=sess)
        epochs_training_Accuracy.append(training_accuracy)         
        if epoch % display_step == 0:
            s = 'accuracy epoch '+str(epoch+1)+': '+str(training_accuracy)    
            if f:
                print(s, end="\n", file=f)     
            print (s)
        f.flush()
        os.fsync(f.fileno())
        
        epochs_loss.append(epoch_loss)
        train.reset_epoch()
    totalLoss= avg_cost
#    print("Optimization Finished!")
    # Test model    
    batch_x, batch_y = test.next_batch(len(test.word))
    feed_dict={x: batch_x, y: batch_y}
    test_accuracy =  accuracy.eval(feed_dict=feed_dict, session=sess)         
    training_accuracy = sum(epochs_training_Accuracy)/len(epochs_training_Accuracy)

    # close session
    sess.close()
    
    s = 'training accuracy: '+str(training_accuracy)+' test accuracy: '+str(test_accuracy) + \
        ' difference (training-test): '+str(test_accuracy - training_accuracy) + \
        ' global loss: '+str(totalLoss)
    if f:
        print(s, end="\n", file=f)     
    print (s)
    f.flush()
    os.fsync(f.fileno())
        # num2str in scientific notation
    epsilon = '%.3e' % epsilon
    # composing stringName
    s = filename+'h1_'+str(n_hidden_1)+'h2_'+str(n_hidden_2)+'h3_'+str(n_hidden3)+ \
        'learnRate_'+str(learning_rate)+'epochs_'+str(training_epochs)+ \
        'batchs_'+str(batch_size)+'b1_'+str(beta1)+'b2_'+str(beta2)+'ep_'+str(epsilon)
        
    line = [learning_rate,training_epochs, batch_size,n_hidden_1, n_hidden_2,n_hidden_3,
            n_input,n_classes,beta1,beta2,epsilon,
            totalLoss,training_accuracy, test_accuracy,test_accuracy-training_accuracy]
    SaveResults(folder+filename+'.csv', line)
    
    test_accuracy = [test_accuracy for i in range(len(epochs_training_Accuracy))]
    CreateFigure(s, epochs_training_Accuracy,test_accuracy, epochs_loss, folder)
    
class DataSet:
    def __init__(self, words):
        w2v = Word2Bigrams()
        p2v = Pos2Vec()
        
        self._index_in_epoch = 0
        self.input = [] # store the input vector
        self.output =[] # store the output vector
        self.word = []  # store the word
        self.pos = []   # store the pos of the word
        
        for w in words:
            vec = w2v.Word2Vec(w)
            # check if it is a valid word
            if vec:
                self.input.append(vec)
                pos = p2v.words[w]
                self.output.append(p2v.Pos2Vec(pos))
                self.word.append(w)
                self.pos.append(pos)
        self.input = np.array(self.input)
        self.output = np.array(self.output)
    
    def reset_epoch (self):
        self._index_in_epoch = 0
        
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.input[start:end], self.output[start:end]

    def next_batch_extended(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.input[start:end], self.output[start:end], self.word[start:end], self.pos[start:end]

#==============================================================================
# scpirt 
#==============================================================================
# folder to save data
try:
    folder = sys.argv[1] 
except IndexError:
    folder = ""
# try to open file for save data
try:
    f = open(sys.argv[2], 'a')
except IndexError:
    f = None
    
# words for feeding the nn
words = list(p2v.words.keys())

# testing, only the first 5000 words
words = words[:25000]
# validating words
words = [w for w in words if w2v._ValidateWord(w)]
s = 'total words:'+str(len(words))
if f:
    print(s, end="\n", file=f)     
print (s)

random.shuffle(words)

# dataset dimension
trainDim = int(trainDim*len(words))   
testDim = int(testDim*len(words))
validDim = int(validDim*len(words))    
# words lists    
train = words[-trainDim:]    

s = 'training words:'+str(len(train))
if f:
    print(s, end="\n", file=f)     
print (s)

test = words[:testDim]
s = 'test words:'+str(len(test))
if f:
    print(s, end="\n", file=f)     
print (s)

f.flush()
os.fsync(f.fileno())

validate = words[testDim:trainDim]    
# dataset creation    
train = DataSet(train)
test = DataSet(test)
validate = DataSet(validate)

# total batch
total_batch = int(len(train.word)/batch_size)

TestingNNs(f)

# if I direct the output into file, I close the file pointer
try:
    f.close()
except:
    pass