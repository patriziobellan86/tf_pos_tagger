# -*- coding: utf-8 -*-

import CreateBigrams
import numpy as np


import tensorflow as tf


bi = CreateBigrams.CreateBigrams()

inData=bi.LoadVectorss()
print ('in loaded')
outData=bi.LoadVectorsOut()
print ('out loaded')

print (len(inData), len(outData),len(inData)==len(outData))
print (inData[0])
print (type(inData))

dimensionInputLayer = len(inData[0])
dimensionOutLayer = 1#len(outData[1])

print ('in out')
print (dimensionInputLayer, dimensionOutLayer)


train = 0.35
test = 1- train  # inutile ma chiarificativa

trainVectorIn=inData[:int(train*len(inData))]
trainVector=outData[:int(train*len(outData))]

testVectorsIn=inData[int(train*len(inData)):]

testVectorsOut=outData[int(train*len(outData)):]




model_path = "model.ckpt"



# config
batch_size = 100 # int(train*len(inData)) / 
learning_rate = 0.01
training_epochs = 10


batch_index= 0
def next_batch (index):
    global batch_index 

    batch_index +=1
    yield inData[batch_index-1:batch_index], outData[batch_index-1:batch_index]

# reset everything to rerun in jupyter
tf.reset_default_graph()



# input images
# None -> batch size can be any size, 784 -> flattened mnist image
x = tf.placeholder(tf.float32, shape=[None, dimensionInputLayer], name="x-input") 
# target 10 output classes
y_ = tf.placeholder(tf.float32, shape=[None, dimensionOutLayer], name="y-input")

# model parameters will change during training so we use tf.Variable
W = tf.Variable(tf.zeros([dimensionInputLayer, dimensionOutLayer]))

# bias
b = tf.Variable(tf.zeros([dimensionOutLayer]))

# implement model
# y is our prediction
y = tf.nn.softmax(tf.matmul(x,W) + b)

# specify cost function
# this is our cost
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# specify optimizer
# optimizer is an "operation" which we can execute in a session
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
  # variables need to be initialized before we can use them
  sess.run(tf.global_variables_initializer())

  # perform training cycles
  for epoch in range(training_epochs):
        
    # number of batches in one epoch
    batch_count = int(int(train*len(inData))/batch_size)
        
    for i in range(batch_count):
      batch_x, batch_y = next_batch(i)
            
      # perform the operations we defined earlier on batch
      sess.run([train_op], feed_dict={x: batch_x, y_: batch_y})
            
    if epoch % 2 == 0: 
      print ("Epoch: ", epoch )
  print ("Accuracy: ", accuracy.eval(feed_dict={x:testVectorsIn, y_: testVectorsOut}))
  print ("done")

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
  

# Save model weights to disk
save_path = saver.save(sess, model_path)
print("Model saved")
