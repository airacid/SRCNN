import tensorflow as tf

import time
import os
import matplotlib.pyplot as plt

import numpy as np
from utils import read_data, input_setup, imsave, merge


class SRCNN(object):
	def __init__(self, image_size=33,
				 label_size=21, batch_size=128, c_dim=1, checkpoint_dir=None,
				 sample_dir=None, FLAGS=None):

		self.image_size = image_size
		self.label_size = label_size
		self.batch_size = batch_size
		self.c_dim = c_dim
		self.checkpoint_dir = checkpoint_dir
		self.sample_dir = sample_dir
		self.FLAGS = FLAGS

	def call(self):
		# input placeholders
		self.images = tf.placeholder(tf.float32, [None, self.image_size,
												  self.image_size, self.c_dim],
									 name='images')
		self.labels = tf.placeholder(tf.float32, [None, self.label_size,
												  self.label_size, self.c_dim],
									 name='labels')

		# build your model

		self.conv1 = self.conv2d(self.images, 64, 9, name='conv1')
		self.conv2 = self.conv2d(self.conv1, 32, 1, name='conv2')
		self.pred = self.conv2d(self.conv2, 1, 5, activation=None, name='pred')

	def train(self):
		print("Training...")
		input_setup(self.FLAGS)
		self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
		tf.summary.scalar('loss', self.loss)
		# preprocess
		data_dir = os.path.join('{}'.format(self.FLAGS.checkpoint_dir),
								"train.h5")
		train_data, train_label = read_data(data_dir)

		self.global_step = tf.train.get_or_create_global_step()
		tf.summary.scalar('global step', self.global_step)
		learning_rate = tf.train.exponential_decay(self.FLAGS.learning_rate,
												   self.global_step,
												   20000, 0.96,
												   staircase=True)
		tf.summary.scalar('loss/learning_rate',learning_rate)
		train_op = tf.train.GradientDescentOptimizer(
			learning_rate).minimize(self.loss,
											   global_step=self.global_step)
		ckpt_dir = self.FLAGS.checkpoint_dir
		with tf.train.MonitoredTrainingSession(
				checkpoint_dir=ckpt_dir) as sess:
			for epoch_idx in range(self.FLAGS.epoch):
				batch_idxs = len(train_data) // self.batch_size
				for idx in range(0, batch_idxs):
					batch_images = train_data[idx * self.batch_size: (
																				 idx + 1) * self.batch_size]
					batch_labels = train_label[idx * self.batch_size: (
																				  idx + 1) * self.batch_size]
					sess.run(train_op, feed_dict={self.images: batch_images,
												  self.labels: batch_labels})

	def inference(self):
		print("Testing...")
		nx, ny = input_setup(self.FLAGS)
		data_dir = os.path.join('{}'.format(self.FLAGS.checkpoint_dir),
								"test.h5")
		saver = tf.train.Saver()
		train_data, train_label = read_data(data_dir)
		with tf.Session() as sess:
			saver.restore(sess,
						  self.FLAGS.checkpoint_dir+'/model.ckpt-315000')
			result = self.pred.eval({self.images: train_data, self.labels: train_label})

		result = merge(result, [nx, ny])
		result = result.squeeze()
		image_path = os.path.join(os.getcwd(), self.FLAGS.sample_dir)
		image_path = os.path.join(image_path, "test_image.png")
		imsave(result, image_path)

	# small blocks

	def conv2d(self, inputs,
			   filters,
			   kernel_size,
			   strides=(1, 1),
			   padding='VALID',
			   data_format='channels_last',
			   dilation_rate=(1, 1),
			   activation=tf.nn.relu,
			   kernel_initializer=tf.random_normal_initializer(stddev=1e-3),
			   bias_initializer=tf.zeros_initializer(),
			   kernel_regularizer=None,
			   bias_regularizer=None,
			   activity_regularizer=None,
			   kernel_constraint=None,
			   bias_constraint=None,
			   name=None):
		return tf.layers.conv2d(inputs=inputs,
								filters=filters,
								kernel_size=kernel_size,
								strides=strides,
								padding=padding,
								data_format=data_format,
								dilation_rate=dilation_rate,
								activation=activation,
								kernel_initializer=kernel_initializer,
								bias_initializer=bias_initializer,
								kernel_regularizer=kernel_regularizer,
								bias_regularizer=bias_regularizer,
								activity_regularizer=activity_regularizer,
								kernel_constraint=kernel_constraint,
								bias_constraint=bias_constraint,
								name=name)
