### Ensemble models ###

import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import Cython

import torch
from torch.autograd import Variable

from helpers import *
from data import create_csv_submission

torch.manual_seed(1)

### Step 1: Load the tweets ###

DATA_PATH = '../twitter-datasets/'
MODEL_PATH = '../models/'

TRAIN_NEG_PATH = os.path.join(DATA_PATH, 'train_neg_full.txt') # 2'500'000 negative tweets
TRAIN_POS_PATH = os.path.join(DATA_PATH, 'train_pos_full.txt') # 2'500'000 positive tweets
TEST_PATH = os.path.join(DATA_PATH, 'test_data.txt')

print("Loading datasets...")
x_text_train, y_train_full = load_data_and_labels(TRAIN_POS_PATH, TRAIN_NEG_PATH)
x_text_test = load_test_data(TEST_PATH)
print("Datasets loaded!")
