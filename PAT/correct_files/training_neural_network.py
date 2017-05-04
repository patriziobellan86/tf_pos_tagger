#!/opt/python/bin/python3


from __future__ import print_function



import argparse

import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes

from Word2Vec import Word2Bigrams, Pos2Vec




w2v = Word2Bigrams()
p2v = Pos2Vec()

#==============================================================================
#  default Parameters
#==============================================================================
# General Parametes

# filename model
model_ext = ".ckpt"

trainDim = 0.7
testDim = 0.2
validDim = 0.1

# Training Parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 50
display_step = 1

# Network Parameters
n_hidden_1 = 500#50 # 1st layer number of features
n_hidden_2 = 250#500 # 2nd layer number of features
n_input = len(w2v.bidict)
n_classes = len(p2v.posdict)

# optimazer params
beta1=0.9
beta2=0.999
epsilon=1e-08


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
            print ('dataset element: ',w)
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


# words for feeding the nn
words = list(p2v.words.keys())

# testing, only the first 5000 words
words = words[:5000]

random.shuffle(words)

# dataset dimension
trainDim = int(trainDim*len(words))   
testDim = int(testDim*len(words))
validDim = int(validDim*len(words))    
# words lists    
train = words[-trainDim:]    
test = words[:testDim]
validate = words[testDim:trainDim]    
# dataset creation    
train = DataSet(train)
test = DataSet(test)
validate = DataSet(validate)

# total batch
total_batch = int(len(train.word)/batch_size)

#==============================================================================
# Network definition
#==============================================================================

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # sparse_softmax_cross_entropy_with_logits
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1, beta2=beta2,epsilon=epsilon).minimize(cost)

correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


## Launch the graph

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()    
sess.run(init)

# Training cycle
lines=[]     # store lines for plot
linesvalidate = []
for epoch in range(training_epochs):
    print(epoch)
    dataplot = []
    avg_cost = 0.
    print('total batch', total_batch)
    # Loop over all batches
    for i in range(total_batch-1):
        batch_x, batch_y = train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
#        feed_dict={x: batch_x, y: batch_y}
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        print ('c',c, 'c/tbatch',c / total_batch, 'avg_cost', avg_cost)
        # Compute average loss
        avg_cost += c / total_batch
        dataplot.append(avg_cost)
    
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", \
            "{:.9f}".format(avg_cost))
        
        batch_x, batch_y = validate.next_batch(10)
        feed_dict={x: batch_x, y: batch_y}
        acc = accuracy.eval(feed_dict=feed_dict, session=sess)
        linesvalidate.append(acc)         
        print('accuracy epoch', acc)    

    lines.append(dataplot)

    train.reset_epoch()

plt.plot(lines)
plt.show()


plt.plot(linesvalidate)
plt.show()  
 
print("Optimization Finished!")



# Test model


#
#correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
## Calculate accuracy
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
  
# Save model weights to disk
save_path = saver.save(sess, 'model'+model_ext)
print("Model saved")


batch_x, batch_y = test.next_batch(len(test.word))
feed_dict={x: batch_x, y: batch_y}
         
print('accuracy', accuracy.eval(feed_dict=feed_dict, session=sess))    

#==============================================================================
# # MAKE PREDICTIONS
#==============================================================================
print ('prediction')

tot = 0
true = 0
validate.reset_epoch()

batch_x, batch_y, word, posWord = validate.next_batch_extended(1)

feed_dict={x: batch_x}
# make a single prediction
predictions = sess.run(pred, feed_dict=feed_dict)
#    prediction=tf.argmax(y,1)
print (predictions)
p=predictions.tolist()
p=p[0]
posPredicted = p2v.PosFromIndex(p.index(max(p)))
print (posPredicted)
print ('prediction for word',word,posWord, 'correct: ', posWord[0] == posPredicted)
tot += 1

if posWord == posPredicted:
    true += 1
for i in range(len(validate.word)-5):
    batch_x, batch_y, word, posWord = validate.next_batch_extended(1)
    
    feed_dict={x: batch_x}
    
    predictions = sess.run(pred, feed_dict=feed_dict)
#    prediction=tf.argmax(y,1)
    print (predictions)
    p=predictions.tolist()
    p=p[0]
    posPredicted = p2v.PosFromIndex(p.index(max(p)))
    print (posPredicted)
    print ('prediction for word',word,posWord, 'correct: ', posWord[0] == posPredicted)

    tot += 1
    if posWord == posPredicted:
        true += 1
        
print (true / tot)
print ('End')





#
#
## find predictions on val set
#    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
#    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
#    
#    predict = tf.argmax(output_layer, 1)
#    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})