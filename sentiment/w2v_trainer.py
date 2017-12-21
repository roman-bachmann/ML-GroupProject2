import os
import numpy as np
import gensim
import Cython

from helpers import *


### Step 1: Load the tweets ###

DATA_PATH = '../twitter-datasets/'
MODEL_PATH = '../models/'

TRAIN_NEG_PATH = os.path.join(DATA_PATH, 'train_neg_full.txt') # 1'250'000 negative tweets
TRAIN_POS_PATH = os.path.join(DATA_PATH, 'train_pos_full.txt') # 1'250'000 positive tweets
TEST_PATH = os.path.join(DATA_PATH, 'test_data.txt')

print("Loading datasets...")
x_text_train, y_train_full = load_data_and_labels(TRAIN_POS_PATH, TRAIN_NEG_PATH)
x_text_test = load_test_data(TEST_PATH)
print("Datasets loaded!")


### Step 2: Build word2vec model ###

vector_length = 100
print("Beginning word2vec training...")
# gensim's word2vec implementation is non-deterministic if using multiple workers.
# Set workers to 1 to make deterministic!
w2v_model = gensim.models.Word2Vec(x_text_train + x_text_test, min_count=1, workers=4, size=vector_length)

# When training finished delete the training model but retain the word vectors:
word_vectors = w2v_model.wv
del w2v_model

word_vectors.save(MODEL_PATH + 'twitter_word_vectors.bin')
print("Training done. Model saved.")
