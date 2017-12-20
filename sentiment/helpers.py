import numpy as np
import re
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def clean_str(string):
    """
    Further tokenization and string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Keyword arguments:
    string -- the string to clean
    """
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

def load_data_and_labels(train_positive_path, train_negative_path):
    """
    Loads tweet data from files, splits the data into words and generates labels.
    Returns split sentences and labels.

    Keyword arguments:
    train_positive_path -- path to positive training tweets
    train_negative_path -- path to negative training tweets
    """
    # Load data from files
    positive_examples = list(open(train_positive_path).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(train_negative_path).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text_train = positive_examples + negative_examples
    x_text_train = [clean_str(sent) for sent in x_text_train]
    x_text_train = [s.split(" ") for s in x_text_train]
    # Generate labels
    #positive_labels = [[0, 1] for _ in positive_examples]
    #negative_labels = [[1, 0] for _ in negative_examples]
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text_train, y]

def load_test_data(test_path):
    """
    Loads test data and splits the data into words.
    Returns split sentences.

    Keyword arguments:
    test_path -- path to test tweets
    """
    # Load data from files
    x_text_test = list(open(test_path).readlines())
    x_text_test = [s.strip() for s in x_text_test]
    # Split by words
    x_text_test = [clean_str(sent) for sent in x_text_test]
    x_text_test = [s.split(" ") for s in x_text_test]
    # Delete the index and comma
    for x in x_text_test:
        del x[0]
        del x[0]
    return x_text_test

def pad_sentences(sentences, padding_word="<PAD/>", sequence_length=0):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.

    Keyword arguments:
    sentences -- all tweets to pad to same length
    padding_word -- word to pad every sentence to the same length (default: <PAD/>)
    sequence_length -- Predefine the length that all sentences should have. Leave
    at 0 to calculate maximum length automatically. (default: 0)
    """
    if sequence_length <= 0:
        sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

class ListDataset(Dataset):
    """Dataset wrapping data and target lists.

    Each sample will be retrieved by indexing both lists along the first
    dimension.

    Arguments:
        data_list (python list): contains sample data.
        target_list (python list): contains sample targets (labels).
    """

    def __init__(self, data_list, target_list):
        assert len(data_list) == len(target_list)
        self.data_list = data_list
        self.target_list = target_list

    def __getitem__(self, index):
        return self.data_list[index], self.target_list[index]

    def __len__(self):
        return len(self.data_list)

def get_tweets_tensor(tweets, word_vectors, vector_length, indices=[]):
    '''Mapping every word to a vector from word2vec
    Padding words are mapped to zero
    Leave indices empty to map every tweet in tweets
    '''

    nb_tweets = len(tweets) if len(indices)==0 else len(indices)
    tweets_vec = np.zeros((nb_tweets, len(tweets[0]), vector_length), dtype=np.float32)

    if indices == []:
        for idx_t, tweet in enumerate(tweets):
            for idx_w, word in enumerate(tweet):
                if word != '<PAD/>':
                    tweets_vec[idx_t, idx_w] = word_vectors.wv[word]
    else:
        for idx_t, orig_idx in enumerate(indices):
            for idx_w, word in enumerate(tweets[orig_idx]):
                if word != '<PAD/>':
                    tweets_vec[idx_t, idx_w] = word_vectors.wv[word]

    return torch.from_numpy(tweets_vec)

def get_tweets_tensor_tfidf(tweets, word_vectors, vector_length, tfidf, indices=[]):
    '''Mapping every word to a vector from word2vec
    Padding words are mapped to zero
    Leave indices empty to map every tweet in tweets
    '''

    nb_tweets = len(tweets) if len(indices)==0 else len(indices)
    tweets_vec = np.zeros((nb_tweets, len(tweets[0]), vector_length), dtype=np.float32)

    if indices == []:
        for idx_t, tweet in enumerate(tweets):
            for idx_w, word in enumerate(tweet):
                if word != '<PAD/>':
                    tweets_vec[idx_t, idx_w] = word_vectors.wv[word] * tfidf[word]
    else:
        for idx_t, orig_idx in enumerate(indices):
            for idx_w, word in enumerate(tweets[orig_idx]):
                if word != '<PAD/>':
                    tweets_vec[idx_t, idx_w] = word_vectors.wv[word] * tfidf[word]

    return torch.from_numpy(tweets_vec)

class CNN(nn.Module):
    """
    Two layer convolutional network with batch normaliztation,
    ReLu activation and dropout after each convolutional layer.

    Keyword arguments:
    S -- Number of words in one sentence (eg 50)
    V -- Length of word vectors (eg 300)
    K -- Kernel width (K*V)
    C -- Number of convolutional filters
    F -- Number of output neurons in fully connected layer
    """
    def __init__(self, S, V, K, C, F):
        super(CNN, self).__init__()

        self.bn = nn.BatchNorm1d(C)         # batch normalization
        self.relu = nn.ReLU()               # ReLU activation
        self.dropout= nn.Dropout(p=0.2)     # dropout layer

        self.conv1 = nn.Conv2d(             # input shape (1, S, V)
                in_channels=1,              # input channels
                out_channels=C,             # number of filters
                kernel_size=(K,V),          # filter size
                padding=(K//2,0)            # to keep size S
        )                                   # output shape (C, S, 1)
        self.max_pool1 = nn.MaxPool1d(2)    # max-pool each filter into S/2 output

        self.conv2 = nn.Conv1d(             # input shape (C, S/2)
                in_channels=C,              # input channels
                out_channels=C,             # number of filters (one for each input channel)
                kernel_size=K,              # filter size
                padding=K//2                # to keep size S/2
        )                                   # output shape (C, S/2)
        self.max_pool2 = nn.MaxPool1d(S//2) # max pool each filter into 1 output

        self.out = nn.Linear(C, F)          # fully connected layer, output F classes
        self.softmax = nn.Softmax(dim=1)    # softmax activation function

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.conv1(out).squeeze(3)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool2(out).squeeze(2)
        out = self.out(out)
        out = self.softmax(out)
        return out
