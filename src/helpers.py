import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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

def accuracy(y_pred, y_true, verbose=False):
    nb_equal = (y_pred == y_true).sum()
    accuracy = nb_equal/len(y_true)
    if verbose:
        print('Accuracy: {}%'.format(accuracy))
    return accuracy
