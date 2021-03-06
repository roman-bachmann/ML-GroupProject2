### Sentiment Model 2 ###
# 2 convolutional layers, kernel size 3, 265 convolutional channels
# Only using word2vec

import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gensim
import Cython

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader

from helpers import *
from data import create_csv_submission


torch.manual_seed(2)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2)
# CUDNN has non-deterministic kernels. Uncomment to make deterministic (but slow):
#torch.backends.cudnn.enabled = False


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


### Step 2: Load word2vec vocabulary ###

vector_length = 100
print("Loading word2vec model...")
word_vectors = gensim.models.keyedvectors.KeyedVectors.load(MODEL_PATH + 'twitter_word_vectors.bin')
print("word2vec model loaded!")


### Step 3: Convert tweets into sentences of vectors ###
print("Converting tweets into sentences of vectors...")

# Compute the number of words of the longest tweet to get the maximal sentence length
sequence_length_train = max(len(x) for x in x_text_train)
sequence_length_test = max(len(x) for x in x_text_test)
sequence_length = max(sequence_length_train, sequence_length_test)
print('Maximum sequence length of train and test data:', sequence_length)

x_text_train_pad = pad_sentences(x_text_train, padding_word="<PAD/>", sequence_length=sequence_length)
x_text_test_pad = pad_sentences(x_text_test, padding_word="<PAD/>", sequence_length=sequence_length)

del x_text_train
del x_text_test

# Split into training and validation data
x_train, x_val, y_train, y_val = train_test_split(x_text_train_pad, y_train_full, test_size=0.01, random_state=42)


### Step 4: Classification ###
print("Setting up classification...")

# Hyper Parameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

train_dataset = ListDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# Network hyperparameters
N = len(train_loader.dataset.data_list)      # Number of tweets (eg 200000)
S = len(train_loader.dataset.data_list[0])   # Number of words in one sentence (eg 50)
V = vector_length                            # Length of word vectors (eg 100)
K = 3                                        # Kernel width (K*V)
C = 256                                      # Number of convolutional filters
F = 2                                        # Number of output neurons in fully connected layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(             # input shape (1, S, V)
                in_channels=1,              # input channels
                out_channels=C,             # number of filters
                kernel_size=(K,V),          # filter size
                padding=(1,0)               # to keep size S
        )                                   # output shape (C, S, 1)
        self.relu = nn.ReLU()               # ReLU activation
        self.max_pool1 = nn.MaxPool1d(2)    # max-pool each filter into S/2 output
        self.conv2 = nn.Conv1d(             # input shape (C, S/2)
                in_channels=C,              # input channels
                out_channels=C,             # number of filters (one for each input channel)
                kernel_size=K,              # filter size
                padding=1                   # to keep size S/2
        )                                   # output shape (C, S/2)
        self.max_pool2 = nn.MaxPool1d(S//2) # max pool each filter into 1 output
        self.out = nn.Linear(C, F)          # fully connected layer, output F classes

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.conv1(out)
        out = self.relu(out).squeeze(3)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.max_pool2(out).squeeze(2).float()
        out = self.out(out)
        return out

cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = nn.CrossEntropyLoss()

cnn.train()
print("Starting training...")
for epoch in range(num_epochs):  # loop over the dataset multiple times
    for i, batch_indices in enumerate(train_loader.batch_sampler):   # iterate over mini-batches
        # Converting tweets to vectors and storing it in a variable
        sentences = get_tweets_tensor(train_loader.dataset.data_list, word_vectors, vector_length, batch_indices)
        if torch.cuda.is_available():
            sentences = sentences.cuda()
        x = Variable(sentences)

        # Converting labels to a variable
        labels = torch.from_numpy(train_loader.dataset.target_list[batch_indices])
        if torch.cuda.is_available():
            labels = labels.cuda()
        y = Variable(labels, requires_grad=False)

        # Forward + Backward + Optimize
        optimizer.zero_grad() # reset gradient
        outputs = cnn(x) # cnn output
        loss = loss_func(outputs, y) # clear gradients for this training step
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients

        if (i+1) % 1000 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

print('Finished Training')

# Evaluate accuracy of predictions from validation data
cnn.eval()
accuracy = 0
nb_steps = 0
step_size = 100 # Calculate in steps, since GPU memory might be too small for whole testset
for i in range(0, len(x_val), step_size):
    val_output = cnn(Variable(get_tweets_tensor(x_val[i:i+step_size], word_vectors, vector_length).cuda()))
    y_val_pred = torch.max(val_output.cpu(), 1)[1].data.numpy().squeeze()
    accuracy += accuracy_score(y_val[i:i+step_size], y_val_pred)
    nb_steps += 1

print('Validation accuracy:', accuracy/nb_steps)

### Step 5: Make predictions and save model ###

cnn.cpu()
torch.save(cnn, './sentiment_models/sentiment_v2.pth')

test_output = cnn(Variable(get_tweets_tensor(x_text_test_pad, word_vectors, vector_length)))
y_pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
y_pred[y_pred == 0] = -1
ids = np.arange(len(y_pred)+1)[1:]
create_csv_submission(ids, y_pred, 'kaggle_sentiment_v2.csv')
