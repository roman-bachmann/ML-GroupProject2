# ML-GroupProject2

***Authors: Lia Bifano, Filippa Bång, Roman Bachmann***

## Architecture Overview

- **./analysis/**: Folder containing jupyter notebooks for the data analysis part.
- **./ensemble/**: Folder containing the models and the ensembling script for the final submission.
	- **ensemble_models/**: Folder containing pretrained pytorch neural networks for each ensemble file.
	- **data.py**: 
	- **helpers.py**: 
	- **model\*.py**: 
	- **ensemble.py**: 
- **./models/**: Folder where pretrained word2vec word vectors should be placed.
- **./sentiment/**: Folder containing all models created in the iterative improvement process.
	- **sentiment_models/**: Folder containing pretrained pytorch neural networks for each sentiment file.
	- **data.py**: 
	- **helpers.py**: 
	- **sentiment\*.py**: 
- **./twitter-datasets/**: Folder where twitter training and testing data should be placed.

## Required packets

Please run all files using the following packets:

- numpy (1.13.3)
- scikit-learn (0.19.0)
- gensim (3.1.0)
- Cython (0.27.3)
- PyTorch (0.3.0.post4)

If available (to severely speed up the training):

- PyTorch with CUDA enabled
- PyTorch with CUDNN enabled

## How to get final submission

## Description of ensemble.py