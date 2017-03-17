#! /usr/bin/env python
#encoding:utf-8
from __future__ import division

import tensorflow as tf
import numpy as np
import math
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_vec_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_vec_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("output_dir", "", "directory for save train output")

tf.flags.DEFINE_string("vocab_embedding_file", "", "word embedding for vocabulary.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))
print("")

# Data Preparatopn
# ==================================================

def tokenizer(iterator):
	"""Tokenizer generator.
	Args:
		iterator: Input iterator with strings.
	Yields:
		array of tokens per each value in the input.
	"""
	for value in iterator:
		yield value.split(' ')

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
num_train = int(len(y)*(1.0-FLAGS.dev_sample_percentage))
vector_train = data_helpers.load_vector(FLAGS.positive_vec_file, FLAGS.negative_vec_file)
num_validate = len(y)-num_train

W_pretrain = np.loadtxt(FLAGS.vocab_embedding_file)
for i in range(len(W_pretrain)):
	if math.isnan(np.sum(W_pretrain[i])):
		W_pretrain[i] = np.zeros(shape=FLAGS.embedding_dim)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,tokenizer_fn=tokenizer)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(1)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]
x_text = np.array(x_text)[shuffle_indices]
vector_train = vector_train[shuffle_indices]

# Split train/test set
x_train, x_dev = x[:num_train], x[num_train:]
y_train, y_dev = y[:num_train], y[num_train:]
vector_train, vector_validate = vector_train[:num_train], vector_train[num_train:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

test_save_dir=FLAGS.output_dir+'/test_result/'
if not os.path.exists(test_save_dir):
	os.makedirs(test_save_dir)
with open(test_save_dir+'/text_label', 'w') as fout:
	for i in range(num_validate):
		idx='0' if y_dev[i][0]==1 else '1'
		fout.write(x_text[i+num_train]+'\t'+idx+'\n')

# Training
# ==================================================

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
	  allow_soft_placement=FLAGS.allow_soft_placement,
	  log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			sequence_length=x_train.shape[1],
			num_classes=y_train.shape[1],
			num_features=vector_train.shape[1],
			vocab_size=len(vocab_processor.vocabulary_),
			embedding_size=FLAGS.embedding_dim,
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters=FLAGS.num_filters,
			l2_reg_lambda=FLAGS.l2_reg_lambda,
			W_pretrain=W_pretrain,
			)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-4)
		#optimizer = tf.train.RMSPropOptimizer(1e-4)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		# Keep track of gradient values and sparsity (optional)
		grad_summaries = []
		for g, v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
				sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.merge_summary(grad_summaries)

		# Output directory for models and summaries
		out_dir=FLAGS.output_dir
		print("Writing to {}\n".format(out_dir))

		# Summaries for loss and accuracy
		loss_summary = tf.scalar_summary("loss", cnn.loss)
		acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

		# Train Summaries
		train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

		# Dev summaries
		dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
		dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
		dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.all_variables())

		graph_dir = os.path.abspath(os.path.join(out_dir, "graphDef"))
		if not os.path.exists(graph_dir):
			os.makedirs(graph_dir)
		tf.train.write_graph(sess.graph_def, graph_dir, "test.pb")

		# Write vocabulary
		vocab_processor.save(os.path.join(out_dir, "vocab"))

		# Initialize all variables
		init = tf.initialize_variables(tf.all_variables(), name='init_all_vars_op')
		sess.run(init)
		
		def save():
			for variable in tf.trainable_variables():
				tensor = tf.constant(variable.eval())
				tf.assign(variable, tensor, name="nWeights")

		def train_step(x_batch, y_batch, features_batch):
			"""
			A single training step
			"""
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
			  cnn.input_features: features_batch
			}
			_, step, summaries, loss, accuracy= sess.run(
				[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			#print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
			train_summary_writer.add_summary(summaries, step)
		
		def dump_eval(y_label,y_predict):
			tp_count = 0
			fn_count = 0
			fp_count = 0
			tn_count = 0
			for l,p in zip(y_label[:,1],y_predict):
				if l==1 and p==1:
					tp_count += 1
				elif l==0 and p==1:
					fn_count += 1
				elif l==1 and p==0:
					fp_count += 1
				else:
					tn_count += 1
			acc_p = tp_count / (tp_count + fp_count)
			acc_n = tn_count / (tn_count + fn_count)
			acc = (tp_count + tn_count) / (tp_count + fn_count + fp_count + tn_count)
			recall = tp_count / (tp_count + fn_count)
			f1 = 2 * recall * acc / (acc + recall)
			with open(test_save_dir+'/predict', 'w') as fout:
				fout.write('\n'.join([str(i) for i in y_predict]))
			return acc_p,acc_n,acc,recall,f1

		def dev_step(x_batch, y_batch, features_batch, writer=None):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: 1.0,
			  cnn.input_features: features_batch
			}
			step, summaries, loss, accuracy, y_predict = sess.run(
				[global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
				feed_dict)
			acc_p,acc_n,acc,recall,f1 = dump_eval(y_batch,y_predict)
			
			time_str = datetime.datetime.now().isoformat()
			print("Eval:{}: step {}, loss {:g}, acc_p {:g}, acc_n {:g}, acc {:g}, recall {:g}, f1 {:g}".format(time_str, step, loss,acc_p,acc_n,acc,recall,f1))
			if writer:
				writer.add_summary(summaries, step)

		# Generate batches
		batches = data_helpers.batch_iter(
			list(zip(x_train, y_train, vector_train)), FLAGS.batch_size, FLAGS.num_epochs)
		# Training loop. For each batch...
		for batch in batches:
			x_batch, y_batch, features_batch = zip(*batch)
			train_step(x_batch, y_batch, features_batch)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % FLAGS.evaluate_every == 0:
				dev_step(x_dev, y_dev, vector_validate, writer=dev_summary_writer)
			if current_step % FLAGS.checkpoint_every == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
