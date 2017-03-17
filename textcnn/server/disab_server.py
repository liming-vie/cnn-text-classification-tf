#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('gen-py')
sys.path.append('..')
sys.path.append('../../la')
sys.path.append('../../la/gen-py')

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import time
import pandas as pd

from disambiguation import Disambiguation
from disambiguation.ttypes import DisABResponse
from la_client import LAClient

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class DisambiguationHandler:
	'''
		used for predicting query if related to movie
	'''
	def __init__(self, feature_dim, trie_mp, term_frequency, checkpoint_dir, la_port):
		'''
			init vocabulary and restore model
		'''
		self.feature_dim = feature_dim + 3
		self.trie_mp = trie_mp
		self.term_frequency=term_frequency
		# load vocab
		vocab_path = os.path.join(FLAGS.checkpoint_dir,'..',"vocab")
		self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
		# load graph
		checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
		graph = tf.Graph()
		# restore session
		with graph.as_default():
			session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False)
			self.session = tf.Session(config=session_conf)
			with self.session.as_default():
				# Load the saved meta graph and restore variables
				saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
				saver.restore(self.session, checkpoint_file)
				# Get the placeholders from the graph by name
				self.input_x = graph.get_operation_by_name("input_x").outputs[0]
				self.input_features = graph.get_operation_by_name("input_features").outputs[0]
				self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
				self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
				# self.scores = graph.get_operstion_by_name('output/scores').outputs[0]
		# la client
		self.la_client = LAClient(la_port)

	def segment(self, texts):
		'''
			call la segmentation service	
		'''
		return self.la_client.request_multiple([t.encode('utf-8') for t in texts])

	def feature_vec(self, query):
		'''
			use trie tree to find titles, and get coressponding feature vectors
			ret: [[begin, end, title, pattern, vec]]
		'''
		# default with a none title
		query = query.decode('utf-8')
		ret=[[0, -1, '', query, [0.0 for _ in range(self.feature_dim)]]]
		ret[0][-1][1]=len(query)
		for i in range(len(query)):
			mv = self.trie_mp
			for j in range(i, len(query)):
				if query[j] not in mv:
					break
				# end of a title
				if '<END>' in mv[query[j]]:
					title=query[i:j+1]
					vec=[self.term_frequency[title] if title in self.term_frequency else 0.0, len(query), len(title)]
					vec.extend(mv[query[j]]['<END>'])
					ret.append([i, j, title, query.replace(title, 'MOVIE'), vec])
				# move down
				mv = mv[query[j]]
		return ret

	def run(self, query):
		'''
			reveive a query, and run textcnn disambiguation
			ret: [DisABResponse]
		'''
		# get feature vectors
		query=query.strip().lower()
		ret=self.feature_vec(query)
	
		x_text=[q[3] for q in ret]
		features=[q[4] for q in ret]
		
		x_text=self.segment(x_text)
		# process to vocab index
		x_test=np.array(list(self.vocab_processor.transform(x_text)))
		# make into batches
		features=pd.DataFrame(features).values
		batches = data_helpers.batch_iter(list(zip(x_test, features)), 64, 1, shuffle=False)
		# predict
		predictions=[]
		for batch in batches:
			x_test_batch, features_batch = zip(*batch)
			batch_prediction=self.session.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0, self.input_features: features_batch})
			predictions = np.concatenate([predictions, batch_prediction])
		# generate response
#		tmp=[DisABResponse(ret[i][0], ret[i][1], ret[i][2].encode('utf-8'), float(predictions[i])) for i in range(len(predictions))]
#		print len(tmp), tmp
		return [DisABResponse(ret[i][0], ret[i][1], ret[i][2].encode('utf-8'), float(predictions[i])) for i in range(len(predictions))]

def tokenizer(iterator):
	for value in iterator:
		yield value.split(' ')

if __name__ == '__main__':
	# Parameters
	tf.flags.DEFINE_string('term_frequency_file', '../../data/term_frequency', 'For term frequency feature')
	tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from restore model")
	tf.flags.DEFINE_string('title_feature_file', '', 'for title feature vectors')
	tf.flags.DEFINE_integer('port', 55555, 'Default port disambiguation service')
	tf.flags.DEFINE_integer('la_port', 1028, 'Default port for la segmentation service')
	# print parameters
	FLAGS = tf.flags.FLAGS
	FLAGS._parse_flags()
	print("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
	    print("{}={}".format(attr.upper(), value))
	print("")
	# load term frequency
	print 'loading term frequency'
	term_frequency = {}
	with open(FLAGS.term_frequency_file) as fin:
		lines=fin.readlines()
		mx=0
		for line in lines:
			t = int(line.split('\t')[1])
			mx=mx if mx>t else t
		for line in lines:
			ps=line.split('\t')
			term_frequency[ps[0].lower().strip().decode('utf-8')]=float(ps[1])/mx
		del lines
	# load pv and category feature vec
	print 'loading feature vectors'
	feature_vec = {}
	with open(FLAGS.title_feature_file) as fin:
		for line in fin:
			ps=line.strip().split('\t')
			feature_vec[ps[0].strip().lower().decode('utf-8')]=[float(i) for i in ps[1].split(' ')]
	feature_dim=len(feature_vec[ps[0].strip().lower().decode('utf-8')])
	print(feature_dim)
	# build trie tree
	print 'building trie tree'
	trie={}
	for k in feature_vec:
		move=trie
		ori=k
		k=k.strip().lower()
		for i in range(len(k)):
			if k[i] not in move:
				move[k[i]]={}
			move=move[k[i]]
		move['<END>']=feature_vec[ori]
	del feature_vec
	# init
	print 'initializing server handler'
	handler = DisambiguationHandler(feature_dim, trie, term_frequency, FLAGS.checkpoint_dir, FLAGS.la_port)
	# run thrift server
	print 'starting service'
	processor = Disambiguation.Processor(handler)
	transport = TSocket.TServerSocket(port=FLAGS.port)
	tfactory = TTransport.TBufferedTransportFactory()
	pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#	server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
	server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
	server.serve()
