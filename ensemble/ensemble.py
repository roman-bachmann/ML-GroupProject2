### Ensemble models ###

import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint as sp_randint

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

TRAIN_NEG_PATH = os.path.join(DATA_PATH, 'train_neg_full.txt') # 1'250'000 negative tweets
TRAIN_POS_PATH = os.path.join(DATA_PATH, 'train_pos_full.txt') # 1'250'000 positive tweets
TEST_PATH = os.path.join(DATA_PATH, 'test_data.txt')

print("Loading datasets...")
x_text_train, y_train_full = load_data_and_labels(TRAIN_POS_PATH, TRAIN_NEG_PATH)
x_text_test = load_test_data(TEST_PATH)
print("Datasets loaded!")


### Step 2: Compute Tf-idf and load word2vec vocabulary ###

print("Computing Tf-idf of twitter dataset...")
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x for x in x_text_train + x_text_test])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('Vocab size : {}'.format(len(tfidf)))

vector_length = 100
print("Loading word2vec model...")
word_vectors = gensim.models.keyedvectors.KeyedVectors.load(MODEL_PATH + 'twitter_word_vectors.bin')
print("word2vec model loaded!")

# Extract 12'500 random tweets
x_text_train, _, y_train_full, _ = train_test_split(x_text_train, y_train_full, test_size=0.995, random_state=42)
del _

# Maximum tweet length that our network was trained on
sequence_length = 74
print('Maximum sequence length of train and test data:', sequence_length)

# Pad the tweets to the same length
x_text_train_pad = pad_sentences(x_text_train, padding_word="<PAD/>", sequence_length=sequence_length)
x_text_test_pad = pad_sentences(x_text_test, padding_word="<PAD/>", sequence_length=sequence_length)
del x_text_train
del x_text_test

def get_predictions(model, x, word_vectors, vector_length):
    test_output = model(Variable(get_tweets_tensor(x, word_vectors, vector_length)))
    return torch.max(test_output, 1)[1].data.numpy().squeeze()

def get_predictions_tfidf(model, x, word_vectors, vector_length, tfidf):
    test_output = model(Variable(get_tweets_tensor_tfidf(x, word_vectors, vector_length, tfidf)))
    return torch.max(test_output, 1)[1].data.numpy().squeeze()


### Model 1 ###
print('Loading model 1 and computing predictions...')
model1 = torch.load('./ensemble_models/model1.pth')
y_train_pred1 = get_predictions(model1, x_text_train_pad, word_vectors, vector_length)
y_test_pred1 = get_predictions(model1, x_text_test_pad, word_vectors, vector_length)
del model1
print('Predictions computed!')


### Model 2 ###
y_train_pred2 = np.empty((0, len(y_train_pred1)))
y_test_pred2 = np.empty((0, len(y_test_pred1)))

for i in range(10):
    print('Loading model 2.' + str(i) + ' and computing predictions...')
    model2 = torch.load('./ensemble_models/model2.' + str(i) + '.pth')
    y_train_pred2_i = get_predictions(model2, x_text_train_pad, word_vectors, vector_length)
    y_test_pred2_i = get_predictions(model2, x_text_test_pad, word_vectors, vector_length)
    y_train_pred2 = np.vstack([y_train_pred2, y_train_pred2_i])
    y_test_pred2 = np.vstack([y_test_pred2, y_test_pred2_i])
    del model2
    print('Predictions computed!')

# Take the median of all predictions
y_train_pred2 = np.median(y_train_pred2, axis=0)
y_test_pred2 = np.median(y_test_pred2, axis=0)


### Model 3 ###
print('Loading model 3 and computing predictions...')
model3 = torch.load('./ensemble_models/model3.pth')
y_train_pred3 = get_predictions(model3, x_text_train_pad, word_vectors, vector_length)
y_test_pred3 = get_predictions(model3, x_text_test_pad, word_vectors, vector_length)
del model3
print('Predictions computed!')


### Model 4 ###
print('Loading model 4 and computing predictions...')
model4 = torch.load('./ensemble_models/model4.pth')
y_train_pred4 = get_predictions_tfidf(model4, x_text_train_pad, word_vectors, vector_length, tfidf)
y_test_pred4 = get_predictions_tfidf(model4, x_text_test_pad, word_vectors, vector_length, tfidf)
del model4
print('Predictions computed!')


### Model 5 ###
print('Loading model 5 and computing predictions...')
model5 = torch.load('./ensemble_models/model5.pth')
y_train_pred5 = get_predictions_tfidf(model5, x_text_train_pad, word_vectors, vector_length, tfidf)
y_test_pred5 = get_predictions_tfidf(model5, x_text_test_pad, word_vectors, vector_length, tfidf)
del model5
print('Predictions computed!')

### Model 6 ###
print('Loading model 6 and computing predictions...')
model6 = torch.load('./ensemble_models/model6.pth')
y_train_pred6 = get_predictions(model6, x_text_train_pad, word_vectors, vector_length)
y_test_pred6 = get_predictions(model6, x_text_test_pad, word_vectors, vector_length)
del model6
print('Predictions computed!')


### Ensemble ###
clf = RandomForestClassifier(oob_score=True, criterion='entropy', random_state=42)
param_grid = {
    'n_estimators': sp_randint(10, 100),
    'max_depth': sp_randint(10, 90),
    'min_samples_split': sp_randint(100, 1000),
    'random_state': sp_randint(0, 999),
}
rf_grid = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=100, cv=5, random_state=42)

print('Training random forest on all predictions...')
ensemble_train = np.vstack([y_train_pred1, y_train_pred2, y_train_pred3, y_train_pred4, y_train_pred6]).T
rf_grid.fit(ensemble_train, y_train_full)

print('Best hyperparameters:', rf_grid.best_params_)
best_rf = rf_grid.best_estimator_
print('Best random forest oob score:', best_rf.oob_score_)
print('Feature importances:', best_rf.feature_importances_)

ensemble_test = np.vstack([y_test_pred1, y_test_pred2, y_test_pred3, y_test_pred4, y_test_pred6]).T
final_preds = best_rf.predict(ensemble_test)

final_preds[final_preds == 0] = -1
print('Final prediction mean:', final_preds.mean())

create_csv_submission(np.arange(1,10001), final_preds, 'kaggle_ensemble.csv')
print("Kaggle csv saved!")
