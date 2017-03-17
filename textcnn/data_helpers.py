import pandas as pd
import numpy as np
import re
import itertools
import random
from collections import Counter


def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	'''
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()
	'''
	return string

def load_data_and_labels(positive_data_file, negative_data_file, load_all=False):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	positive_examples = []
	fin = open(positive_data_file,'r')
	for line in fin.readlines():
		#llist = line.strip().split(' ')
		positive_examples.append(line.strip())
	fin.close()
	negative_examples = []
	fin = open(negative_data_file,'r')
	for line in fin.readlines():
		#llist = line.strip().split(' ')
		negative_examples.append(line.strip())
	fin.close()
	'''
	if not load_all:
		# make it 1:1
		pl=len(positive_examples)
		nl=len(negative_examples)
		if pl > nl:
			random.shuffle(positive_examples)
			positive_examples=positive_examples[:nl]
		elif pl < nl:
			random.shuffle(negative_examples)
			negative_examples=negative_examples[:pl]
	'''
	# Split by words
	x_text = positive_examples + negative_examples
	#for x in x_text:
	#	 print x
	#x_text = [clean_str(sent) for sent in x_text]
	# Generate labels
	positive_labels = [[0, 1] for _ in positive_examples]
	negative_labels = [[1, 0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)
	return [x_text, y]

def load_vector(positive_vec_file, negative_vec_file):
	positive_examples = []
	fin = open(positive_vec_file,'r')
	for line in fin.readlines():
		#llist = line.strip().split(' ')
		positive_examples.append(map(float, line.strip().split()))
	fin.close()
	negative_examples = []
	fin = open(negative_vec_file,'r')
	for line in fin.readlines():
		#llist = line.strip().split(' ')
		negative_examples.append(map(float, line.strip().split()))
	fin.close()
	# Split by words
	vectors = positive_examples + negative_examples
	df = pd.DataFrame(vectors)
	return df.values

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(len(data)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
