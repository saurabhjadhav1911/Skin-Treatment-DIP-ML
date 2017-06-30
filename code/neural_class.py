from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

class NeuralNetwork():
  def __init__(self,Layer_Neuron_count):

    self.Layer_Neuron_count=tf.constant(Layer_Neuron_count,dtype=tf.int32)
    self.Layer_Numbers=len(Layer_Neuron_count)
    self.Layer_Synapses=[{'weights':tf.Variable(tf.zeros([Layer_Neuron_count[i],Layer_Neuron_count[i+1]])),'biases':tf.Variable(tf.zeros([Layer_Neuron_count[i+1]]))} for i in range(self.Layer_Numbers-1)]
    
  def Forward(self,x):
    y=x
    for layer in self.Layer_Synapses:
      y=tf.matmul(y,layer['weights'])+layer['biases']
      y=tf.nn.relu(y)
    return y

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 3])
y_ = tf.placeholder('float',[None,2])

NN=NeuralNetwork(Layer_Neuron_count=[3,5,2])
mnist = input_data.read_data_sets('', one_hot=True)
def train_neural_network(x):
  prediction = NN.Forward(x)
  # OLD VERSION:
  #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
  # NEW:
  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_) )
  optimizer = tf.train.AdamOptimizer().minimize(cost)
  
  hm_epochs = 10
  with tf.Session() as sess:
      # OLD:
      #sess.run(tf.initialize_all_variables())
      # NEW:
      sess.run(tf.global_variables_initializer())

      for epoch in range(hm_epochs):
          epoch_loss = 0
          for p in range(5000):
              epoch_x, epoch_y = 
              _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y_: epoch_y})
              epoch_loss += c

          print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))

      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
      print('Accuracy:',accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))

train_neural_network(x)