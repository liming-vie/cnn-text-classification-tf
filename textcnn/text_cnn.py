import tensorflow as tf
import numpy as np


class TextCNN(object):
	"""
	A CNN for text classification.
	Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
	"""
	def __init__(
	  self, sequence_length, num_classes, num_features, vocab_size,
	  embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, W_pretrain=None):

		# Placeholders for input, output and dropout
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.input_features = tf.placeholder(tf.float32, [None, num_features], name="input_features")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		#self.dropout_keep_prob = dropout_keep_prob 

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		with tf.device('/cpu:0'), tf.name_scope("embedding"):
			if W_pretrain is None:
				W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
			else:
				W = tf.Variable(W_pretrain,dtype=tf.float32,name="W")
				#W = tf.constant(W_pretrain,dtype=tf.float32,name="W")
			embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					embedded_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				#h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		h_pool = tf.concat(3, pooled_outputs)
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

		with tf.name_scope("full"):
			combine_features = tf.concat(1, [h_pool_flat, self.input_features])
			fc = tf.contrib.layers.fully_connected(combine_features, 256, activation_fn=tf.nn.relu)

		with tf.name_scope("dropout"):
			h_drop = tf.nn.dropout(fc, self.dropout_keep_prob)

		with tf.name_scope("output"):
			W = tf.get_variable(
				"W",
				shape=[256, num_classes],
				initializer=tf.contrib.layers.xavier_initializer())
#			W = tf.Variable(tf.random_uniform([256, num_classes], -1.0, 1.0),name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
		
		# CalculateMean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

		#self.saver = tf.train.Saver(tf.all_variables())	
