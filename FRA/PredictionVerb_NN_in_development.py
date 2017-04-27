'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import csv




import pandas as pd

# TODO

# load data from dataset and create wrapper functions for batching

# input and output dimensions are read from dataset

training = 'training.csv'
test = 'test.csv'
configuration = 'configuration.ini'


# read configuration file
with open (configuration, 'r') as f:
    config = [int(w) for w in [x.strip() for x in f.readlines()]]
inputnode = config[1]
outputnode = config[-1]

def ReadInput (dataline):
    return [dataline[i] for i in range(1,config[1])]

def ReadOutput (dataline):
    return [dataline[i] for i in range ((config[1]+1),len(dataline))] # read out only the last cell that represent the output vector

def LoadCsvDataFrame (filename):
        vectors = {}
        infile  = open(filename, "r")
        reader = csv.reader(infile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            vectors[row[0]] = row[1:]
            vectors[row[0]] = {'inputVector': ReadInput(row), 'outputVector':ReadOutput(row), 'outLabel': row[config[1]]}
            
        return vectors    
a=LoadCsvDataFrame(test)


# ATTENTION!!1 LOST ONE CELL AND I DO NOT UNDERSTAND WHERE! INPUT VECTOR LEN < DI 1

training = LoadCsvDataFrame(training)
test = LoadCsvDataFrame(test)




def FeaturesExtractor (word):
    f1 = word[-1]   # last character
    f2 = word[-2:] # last 2 char
    f3 = word[-3:]
    f4 = word[0]
    f5 = word[:1]
    f6 = word[:2]
    
    return {'f1':f1,'f2':f2,'f3':f3,'f4':f4,'f5':f5,'f6':f6}

w='mangiavo'
print (FeaturesExtractor(w))








import nltk
featuresets = [(FeaturesExtractor(w),training[w]['outLabel'] ) for w in training.keys()]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

#>>> classifier.classify(FeaturesExtractor('mangiavo'))
#'VER'
#>>> classifier.classify(FeaturesExtractor('mangiavamo'))
#'VER'
#>>> classifier.classify(FeaturesExtractor('mangiai'))
#'ADJ'
#>>> classifier.classify(FeaturesExtractor('mangiai'))















# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 =500 # 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = config[1]# 784 # MNIST data input (img shape: 28*28)
n_classes = config[-1] #10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


## continuare da qui con funztione che gestiste batching
## Launch the graph
#with tf.Session() as sess:
#    sess.run(init)
#
#    # Training cycle
#    for epoch in range(training_epochs):
#        avg_cost = 0.
#        total_batch = int(mnist.train.num_examples/batch_size)
#        print('total batch', total_batch)
#        # Loop over all batches
#        for i in range(total_batch):
#            batch_x, batch_y = mnist.train.next_batch(1)#batch_size)
##            with open('batch_x.txt','a') as fx:
##                for i in range(len(batch_x)):
###                print (len(batch_x))
#            print (batch_x[0])
#            print 
#            print 
#            
###                print
####                print (batch_x[1])
###                    with open('batch_y.txt','a') as fy:
###                        fx.write(str(batch_x[1]))
###                        fy.write(str(batch_y[1]))
#            print (len(batch_y[0]))
##                break
##                    
#            # Run optimization op (backprop) and cost op (to get loss value)
#            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
#                                                          y: batch_y})
#            # Compute average loss
#            avg_cost += c / total_batch
#        # Display logs per epoch step
#        if epoch % display_step == 0:
#            print("Epoch:", '%04d' % (epoch+1), "cost=", \
#                "{:.9f}".format(avg_cost))
#    print("Optimization Finished!")
#
#    # Test model
#    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#    # Calculate accuracy
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))