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
import numpy as np

from tensorflow.python.framework import dtypes

training = 'training.csv'
test = 'test.csv'
configuration = 'configuration.ini'

model_path = "model.ckpt"

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 1# # 100
display_step = 1

class DataSet(object):
    def __init__(self, data,dtype=dtypes.float32):
        dtype = dtypes.as_dtype(dtype).base_dtype
        
        self.input = []
        self.output =[]
        for k in data.keys():
            a = [float(w) for w in data[k]['inputVector']]
#            a = np.asarray(a) 
#            a.reshape(364,1)
            assert(len(a)==364)
            self.input.append(a)#tf.constant(a))
            
#            a = [np.array(x, dtype=np.float32) for x in data[k]['outputVector']]
            a = [float(w) for w in data[k]['outputVector']]
#            a = np.asarray(a) 
#            a.reshape(23,1)
            assert(len(a)==23)
            self.output.append(a)#tf.constant(a))            

        self.input = np.array(self.input)
        self.output = np.array(self.output)
            
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self.input[start:end], self.output[start:end]

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

# ATTENTION!!1 LOST ONE CELL AND I DO NOT UNDERSTAND WHERE! INPUT VECTOR LEN < DI 1

training = LoadCsvDataFrame(training)
test = LoadCsvDataFrame(test)


total_batch = int (len(training)/batch_size)  # int(mnist.train.num_examples/batch_size)

training = DataSet(training)
test = DataSet(test)


# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 364#config[1]# 784 # MNIST data input (img shape: 28*28)
n_classes = 23#config[-1] #10 # MNIST total classes (0-9 digits)

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()



correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#tf.SparseTensorValue
## continuare da qui con funztione che gestiste batching
## Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        print(epoch)
        
        avg_cost = 0.
        print('total batch', total_batch)
        # Loop over all batches
        for i in range(total_batch-1):
#            input_fn=lambda: my_input_fn(test_set)
            batch_x, batch_y = training.next_batch(batch_size)
#            batch_y, batch_x = training.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            feed_dict={x: batch_x, y: batch_y}
                        
#            c = sess.run(accuracy, feed_dict=feed_dict)
#            c = sess.run(optimizer, feed_dict)
            _, c = sess.run([optimizer, cost], feed_dict)#={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
      
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved")

