# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:26:22 2017

@author: 5558
"""

import tensorflow as tf
import numpy as np


def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

#sess = tf.Session()
#sess.run(tf.initialize_all_variables())

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")


def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r
    
def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        return tf.matmul(input_data, W, name="matmul") + b
    
keep_prob = tf.placeholder(tf.float32, []) 
x = tf.placeholder(tf.float32,shape=(1,32,10,13))
y = tf.placeholder(tf.float32,[])
lr = tf.placeholder(tf.float32,[])

conv1_1 = conv_relu_batch(x, 128, 3,1, name="conv1_1")
conv1_1_d = tf.nn.dropout(conv1_1, keep_prob)
conv1_2 = conv_relu_batch(conv1_1_d, 128, 3,2, name="conv1_2")
conv1_2_d = tf.nn.dropout(conv1_2, keep_prob)
conv2_1 = conv_relu_batch(conv1_2_d, 256, 3,1, name="conv2_1")
conv2_1_d = tf.nn.dropout(conv2_1, keep_prob)
conv2_2 = conv_relu_batch(conv2_1_d, 256, 3,2, name="conv2_2")
conv2_2_d = tf.nn.dropout(conv2_2, keep_prob)
conv3_1 = conv_relu_batch(conv2_2_d, 512, 3,1, name="conv3_1")
conv3_1_d = tf.nn.dropout(conv3_1, keep_prob)
conv3_2= conv_relu_batch(conv3_1_d, 512, 3,1, name="conv3_2")
conv3_2_d = tf.nn.dropout(conv3_2, keep_prob)
conv3_3 = conv_relu_batch(conv3_2_d, 512, 3,2, name="conv3_3")
conv3_3_d = tf.nn.dropout(conv3_3, keep_prob)

dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
flattened = tf.reshape(conv3_3_d, [-1, dim])

fc6 = dense(flattened, 2048, name="fc6")

logits = dense(fc6, 1, name="dense")

#B, W, H, C = 32, 32,32, 9
#train_step = 25000
lrr = 1e-3
#weight_decay = 0.005
drop_out = 0.25
yieldd = 6

loss_err = tf.nn.l2_loss(logits - y)

train_op = tf.train.AdamOptimizer(lr).minimize(loss_err)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#final = final_tensor.reshape(1,32,10,13)

#for i in range(500):
 #     _, train_loss = sess.run([train_op, loss_err], feed_dict={x:final,y:yieldd,lr:lrr,keep_prob: drop_out})
  #    print(train_loss)














