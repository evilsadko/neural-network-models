#-*- coding: utf-8 -*-
__author__ = "skynet"
import time
import tensorflow as tf
import os
import datetime
import numpy as np
import cv2
import common
DIGITS = u'0123456789'
LETTERSBRU = u'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
LETTERSB = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERSRU = u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
CHARS = DIGITS + LETTERSB + ' '
train=True
num_classes = len(CHARS) + 1
# Utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')

class CNNLSTM(object):
	def __init__(self, mode):
		self.mode = mode
		# image
		self.inputs = tf.placeholder(tf.float32, [None, 64, 128, 1])#[None, 60, 180, 1])
                #tf.placeholder(tf.float32,[None,None,None])# [None,64,128,1]) # 3 | 1?
		# SparseTensor required by ctc_loss op
		self.labels = tf.sparse_placeholder(tf.int32)
		# 1d array of size [batch_size]
		self.seq_len = tf.placeholder(tf.int32, [None])
		# l2
		self._extra_train_ops = []
        def build_graph(self):
                self._build_model()
                self.merged_summay = tf.summary.merge_all()
        
        def _build_model(self):
		filters = [64, 128, 128, 64] 
		strides = [1, 2]
                
		with tf.variable_scope('cnn'):
                    #x_expanded = tf.expand_dims(self.inputs, 1)
		    # 1
		    W_conv1 = weight_variable([7, 7, 1, 32]) #[3, 3, 1, 48] v2 -> [7, 7, 1, 12] #60x180
		    b_conv1 = bias_variable([32]) # 12
		    h_conv1 = tf.nn.relu(conv2d(self.inputs, W_conv1) + b_conv1)
		    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))      #60x180>30x90

		    # 2
		    W_conv2 = weight_variable([5, 5, 32, 64]) # [5, 5, 48, 64] | [5, 5, 12, 24]
		    b_conv2 = bias_variable([64]) # 24
		    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))  #30x90>15x45

		    # 3
		    W_conv3 = weight_variable([3, 3, 64, 64]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv3 = bias_variable([64]) #[128] | 48
		    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
		    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2)) #15x45>7,5x22,5
		    # 4
		    W_conv4 = weight_variable([3, 3, 64, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv4 = bias_variable([128]) #[128] | 48
		    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
		    h_pool4 = max_pool(h_conv4, ksize=(2, 2), stride=(2, 2)) #7,5x22,5>3,75x11,25
                    # 5
		    W_conv5 = weight_variable([3, 3, 128, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv5 = bias_variable([128]) #[128] | 48
		    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
		    h_pool5 = max_pool(h_conv5, ksize=(2, 2), stride=(2, 2)) # [64,524], In[1]: [560,2048]
                    # 6
		    W_conv6 = weight_variable([2, 2, 128, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv6 = bias_variable([128]) #[128] | 48
		    h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
		    h_pool6 = max_pool(h_conv6, ksize=(2, 2), stride=(2, 2)) # [64,524], In[1]: [560,2048]
                    # 7
		    W_conv7 = weight_variable([2, 2, 128, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv7 = bias_variable([128]) #[128] | 48
		    h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
		    h_pool7 = max_pool(h_conv7, ksize=(2, 2), stride=(2, 2)) # [64,524], In[1]: [560,2048]
                    # 8
		    W_conv8 = weight_variable([2, 2, 128, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
		    b_conv8 = bias_variable([128]) #[128] | 48
		    h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
		    h_pool8 = max_pool(h_conv8, ksize=(2, 2), stride=(2, 2)) # [64,524], In[1]: [560,2048]

		with tf.variable_scope('lstm'): 
                    # [batch_size, max_stepsize, num_features]
		    x = tf.reshape(h_pool7, [128, -1, filters[3]])
		    x = tf.transpose(x, [0, 2, 1])  # batch_size * 64 * 48 WTF??? транспарирует матричная операция
		    x.set_shape([128, 64, 2])
		    # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
		    cell = tf.contrib.rnn.LSTMCell(512, state_is_tuple=True) # работает с 180x60 512 rnn сетей/256
		    if self.mode == 'train':
		        cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

		    cell1 = tf.contrib.rnn.LSTMCell(512, state_is_tuple=True)
		    if self.mode == 'train':
		        cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

		    # Stacking rnn cells
		    stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

		    # The second output is the last state and we will not use that
		    outputs, _ = tf.nn.dynamic_rnn(stack, x, self.seq_len, dtype=tf.float32)

		    # Reshaping to apply the same weights over the timesteps
		    outputs = tf.reshape(outputs, [-1, 128])

		    W = tf.get_variable(name='W',
		                        shape=[128, num_classes],
		                        dtype=tf.float32,
		                        initializer=tf.contrib.layers.xavier_initializer())
		    b = tf.get_variable(name='b',
		                        shape=[num_classes],
		                        dtype=tf.float32,
		                        initializer=tf.constant_initializer())

		    self.logits = tf.matmul(outputs, W) + b
		    # Reshaping back to the original shape
		    shape = tf.shape(x)
		    self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
		    # Time major
		    self.logits = tf.transpose(self.logits, (1, 0, 2))

