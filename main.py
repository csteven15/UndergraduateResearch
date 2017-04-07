import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import scipy.misc
import sys
import random
import time
import os

#terminal arguments ==============================

argc = len(sys.argv)
if argc != 3:
	print 'format is main.py /gpu:0 50'

xpu_n = str(sys.argv[1])
epoch = int(sys.argv[2])
batch = 10
h = 256
w = 216

#useful methods

def get_patient_number(i):
    if (i < 10):
        return "0{}".format(i)
    return i

def check_to_create_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#CNN Model =======================================

def static(out,gt):
	TP=np.sum(out[gt==1])
	FP=np.sum(out[gt==0])
	FN=np.sum(gt[out==0])
	TN=np.sum(1-(out[gt==0]))
	sensivity=float(TP)/(TP+FN+0.00001)
	#print "sensitivity=",sens0.84088094  0.25596829  0.00453671  0.0089745 ivity
	specificity=float(TN)/(TN+FP+0.00001)
	#print "specificity=",specificity
	precision=float(TP)/(TP+FP+0.00001)
	#print "precision=",precision
	#print "accuracy =",float(TP+TN)/(TP+FP+TN+FN+0.00001)
	#print "F1_score =",float(2*TP)/(2*TP+FP+FN+0.00001)
	dice = np.sum(TP)*2.0 / (np.sum(gt) + np.sum(out)+0.00001)
	#print "dice=",dice
	return [sensivity,specificity,precision,dice]

def load_data(path):
	original_images = []
	binary_images = []
	for i in range(1, 17):
		p = get_patient_number(i)
		DIR = path + 'P{}/P{}binary/'.format(p, p)
		for s in range(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])):
			file_name = 'P%d-%04d' % (i, s) + '.png'
			original_dir = '{}{}'.format(DIR, file_name)
			binary_dir = '{}{}'.format(DIR, file_name)
			original_image = scipy.misc.imread(original_dir)
			binary_image = scipy.misc.imread(binary_dir)
			original_images.append(original_image)
			binary_images.append(binary_image)
	return np.array(original_images), np.array(binary_images)


directory = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data'
if not os.path.exists(directory):
	os.makedirs(directory)

train_dir = '{}/train/'.format(directory)
test_dir = '{}/test/'.format(directory)
valid_dir = '{}/valid/'.format(directory)

train_x, train_y = load_data(train_dir)
test_x, test_y = load_data(test_dir)

print train_x.shape, train_y.shape

exampleImage = [train_x[0]]
images = np.array([exampleImage])

#Create Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.InteractiveSession(config=config)

with tf.device(xpu_n):

	#prebuilt layers
	def conv_batch_norm_relu(inputs, filters = 1, kernel_size=(1, 1)):
		conv = tf.layers.conv2d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			padding='same',
			data_format='channels_last')
		batch = tf.layers.batch_normalization(
			inputs=conv,
			axis=01)
		relu = tf.nn.relu(batch)
		layer = batch
		return layer
	def max_pooling(inptus, scale):
		return tf.layers.max_pooling2d(
			inputs,
			pool_size=scale,
			strides=scale)
	def up_sampling(inputs, scale):
		shape = inputs.get_shape()
		return tf.image.resize_bilinear(
			images=inputs,
			size=[shape[1].value * scale, shape[2].value * scale])
	def conv_logistic(inputs, filters=1, kernel_size=(1,1)):
		conv = tf.layers.conv2d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			padding='same',
			data_format='channels_last',
			activation=tf.nn.sigmoid
			)
	def loss_function(actual, predicted):
		return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(predicted, actual, pos_weight=10))

	#tensors
	x = tf.placeholder(tf.float32, [None, h, w])
	y_ = tf.placeholder(tf.float32, [None, h, w])

	input_x = tf.reshape(x, [-1, h, w, 1])
	output_y = tf.reshape(y_, [-1, h, w, 1])

	#encoding layers
	layer0_0 = conv_batch_norm_relu(input_x, filters=64, kernel_size=(3,3))
	layer0_1 = conv_batch_norm_relu(layer0_0, filters=64, kernel_size=(3,3))
	layer0_2 = conv_batch_norm_relu(layer0_1, filters=64, kernel_size=(3,3))
	output_0 = max_pooling(layer0_2, scale=2)

	layer1_0 = conv_batch_norm_relu(output_0, filters=128, kernel_size=(3,3))
	layer1_1 = conv_batch_norm_relu(layer1_0, filters=128, kernel_size=(3,3))
	layer1_2 = conv_batch_norm_relu(layer1_1, filters=128, kernel_size=(3,3))
	output_1 = max_pooling(layer1_2, scale=2)

	layer2_0 = conv_batch_norm_relu(output_1, filters=256, kernel_size=(3,3))
	layer2_1 = conv_batch_norm_relu(layer2_0, filters=256, kernel_size=(3,3))
	layer2_2 = conv_batch_norm_relu(layer2_1, filters=256, kernel_size=(3,3))

	#decoding layers
	layer3_0 = conv_batch_norm_relu(layer2_2, filters=256, kernel_size=(3,3))
	layer3_1 = conv_batch_norm_relu(layer3_0, filters=256, kernel_size=(3,3))
	layer3_2 = conv_batch_norm_relu(layer3_1, filters=256, kernel_size=(3,3))

	output_3 = up_sampling(layer3_2, scale=2)
	layer4_0 = conv_batch_norm_relu(output_3, filters=128, kernel_size=(3,3))
	layer4_1 = conv_batch_norm_relu(layer4_0, filters=128, kernel_size=(3,3))
	layer4_2 = conv_batch_norm_relu(layer4_1, filters=128, kernel_size=(3,3))

	output_4 = up_sampling(layer4_2, scale=2)
	layer5_0 = conv_batch_norm_relu(output_4, filters=64, kernel_size=(3,3))
	layer5_1 = conv_batch_norm_relu(layer5_0, filters=64, kernel_size=(3,3))
	layer5_2 = conv_batch_norm_relu(layer5_1, filters=64, kernel_size=(3,3))
	output_x = conv_logistic(layer5_2)

	loss = loss_function(output_x, output_y)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

	#run session
	saver = tf.train.Saver(max_to_keep=0)
	init = tf.global_variables_initializer()
	sess.run(init)

X_train = train_x
Y_train = train_y

print 'begin training'

#training
for epoch_num in range(epoch):
	sh_train = trandom.sample(range(len(X_train)), len(X_train))
	X_train = X_train[sh_train,:]
	Y_train = Y_train[sh_train,:]

	iteration = len(X_train) / batch

	avg_loss = 0

	for i in range(int(iteration)):
		batch_x = X_train[batch*i:(i+1)*batch,:,:]
		batch_y = Y_train[batch*i:(i+1)*batch,:,:]

		train_step.run(feed_dict={x: batch_y, y_: batch_y})

		loss_eval = loss.eval(feed_dict={x: batch_x, y_: batch_y})
		avg_loss = avg_loss + loss_eval

	avg_loss = avg_loss / i

	print 'Epoch: {} -> avg. loss: {}'.format(epoch_num+1, avg_loss)

	model_dir = '{}/models/'.format(directory)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	saver.save(sess, "{}model_{}.ckpt".format(model_dir,epoch_num))

print 'finished training'
