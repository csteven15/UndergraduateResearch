from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#matpy
#pydicom

# Imports
import numpy as np
import tensorflow as tf
import os
#from scipy.misc import imread, imsave, imresize

def cnn_model_fn(features):
	'''model function for cnn'''

	#input layer
	input_layer = tf.reshape(features, [320, 320])

	#size 320 x 320 encoder
	for i in xrange(3):
		#convolution + ReLU
		conv1 = tf.layers.conv2d(inputs = input_layer, filters = 64, kernel_size = [320, 320], padding = "same", activation = none)
		#batch normalization
		for j in xrange(64):
			batch1 = tf.layers.batch_normalization(conv1, axis = 1)
			conv1 = batch1
		input_layer = batch1
	#max pooling layer
	for i in xrange(64):
		max_pool1 = tf.layers.max_pooling2d(inputs = input_layer, pool_size = [160, 160], strides = 2)
		input_layer = max_pool1

	#size 160 x 160 encoder
	for i in xrange(3):
		#convolution + ReLU
		conv2 = tf.layers.conv2d(inputs = input_layer, filters = 128, kernel_size = [160, 160], padding = "same", activation = tf.nn.relu)
		#batch normalization
		for j in xrange(128):
			batch2 = tf.layers.batch_normalization(conv2, axis = 1)
			conv2 - batch2
		input_layer = batch2

	#max pooling layer
	for i in xrange(128):
		max_pool2 = tf.layers.max_pooling2d(inputs = input_layer, pool_size = [80, 80], strides = 2)
		input_layer = max_pool2

	#size 80 x 80 encoder
	for i in xrange(3):
		#convolution + ReLU layer
		conv3 = tf.layers.conv2d(inputs = input_layer, filters = 256, kernel_size = [80, 80], padding = "same", activation = tf.nn.relu)
		#batch normalization
		for j in xrange(256):
			batch3 = tf.layers.batch_normalization(conv3, axis = 1)
			conv3 = batch3
		input_layer = batch3

	#size 80 x 80 decoder
	for i in xrange(3):
		#convolution + ReLU layer
		conv4 = tf.layers.conv2d(inputs = input_layer, filters = 256, kernel_size = [80, 80], padding = "same", activation = tf.nn.relu)
		#batch normalization
		for j in xrange(256):
			batch4 = tf.layers.batch_normalization(conv4, axis = 1)
			conv4 = batch4
		input_layer = batch4

	#size 160 x 160 decoder
	#upsampling layer
	for i in xrange(128):
		upsampling1 = tf.image.resize_images(input_layer, [160, 160])
		input_layer = upsampling1

	for i in xrange(3):
		#convolution + ReLU layer
		conv5 = tf.layers.conv2d(inputs = input_layer, filters = 128, kernel_size = [160, 160], padding = "same", activation = tf.nn.relu)
		#batch normalization
		for j in xrange(128):
			batch5 = tf.layers.batch_normalization(conv5, axis = 1)
			conv5 = batch5
		input_layer = batch5

	#size 320 x 320 decoder
	#upsampling layer
	for i in xrange(64):
		upsampling2 = tf.image.resize_images(input_layer, [320, 320])
		input_layer = upsampling2

	for i in xrange(3):
		#convolution + ReLU layer
		conv6 = tf.layers.conv2d(inputs = input_layer, filters = 128, kernel_size = [160, 160], padding = "same", activation = tf.nn.relu)
		#batch normalization
		for j in xrange(128):
			batch6= tf.layers.batch_normalization(conv6, axis = 1)
			conv6 = batch6
		input_layer = batch6
	final = tf.layers.conv2d(inputs = input_layer, filters = 1, kernel_size = [320, 320], padding = "same", activation = tf.nn.sigmoid)

dir = "/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_segmentation/"
files = []  # create an empty list
for dirName, subdirList, fileList in os.walk(dir):
	for filename in fileList:
		if "dcm" in filename.lower():
			files.append(os.path.join(dirName,filename))
files.sort()
