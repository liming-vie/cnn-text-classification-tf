#! /usr/bin/env python
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("positive_data_file", "/home/work/data/huangliming/baike_click_movie/features/data_query_pos_rs", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "/home/work/data/huangliming/baike_click_movie/features/data_query_neg", "Data source for the positive data.")

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
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file, load_all=True)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length,tokenizer_fn=tokenizer)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
for id in range(len(vocab_processor.vocabulary_)):
	print vocab_processor.vocabulary_.reverse(id)
