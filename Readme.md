# AG News Classification With Bert Model

Multi-label Classification using the BERT (Bidirectional Encoder Representations from Transformers) model. This code demonstrates how to preprocess text data, train a BERT-based multi-label classification model, and evaluate its accuracy. 


## Introduction

This project uses the BERT model to perform classification on news title and text data. It includes data preprocessing, model training, and evaluation steps. Sentiment analysis is the process of determining the sentiment or emotional tone of a piece of text, such as positive, negative, or neutral.

## Prerequisites

Before running the code, you need to ensure that you have the following prerequisites:

    Python 3.7+
    PyTorch
    Hugging Face Transformers library
    pandas
    CUDA-enabled GPU (optional but recommended for faster training)

## Installation

To install the required Python libraries, you can use the following commands:

    pip install torch
    pip install transformers
    pip install pandas

## Usage

### Data Preprocessing

The code performs the following data preprocessing steps:

    Imports necessary libraries and checks for GPU availability.
    Loads training and testing data from CSV files (train.csv and test.csv).
    Combines the "Title" and "Description" columns into a single "Text" column.
    Converts class labels to the correct format (e.g., mapping 1 to 0, 2 to 1, etc.).
    Tokenizes the text data using the BERT tokenizer, "Encode_plus" is used to convert the text into format suitable for Bert model.
    Creates PyTorch Datasets and DataLoaders for training and testing.

### Model Details

The code uses the bert-base-cased pre-trained BERT model and fine-tunes it for text classification with a specified number of output classes (in this case, 4 classes). You can customize the model architecture by changing the BertForSequenceClassification parameters in the script.

### Training

Hyperparameter
    batch_size_train= 64
    batch_size_test= 64
    max_len_train=80
    max_len_test=80
    lr=2e-5
    epochs= 1   

The training section of the code includes the following steps:

    Defines the BERT-based sequence classification model using BertForSequenceClassification.
    Moves the model to the GPU if available.
    Defines the loss function and optimizer.
    Trains the model for a specified number of epochs.
    Prints training accuracy after each epoch.

### Evaluation

The evaluation section of the code includes the following steps:

    Switches the model to evaluation mode.
    Evaluates the model's performance on the testing dataset.
    Prints testing accuracy after each epoch.

## Credits
reference link:
    https://blog.51cto.com/u_15127680/3841198
    https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html?fbclid=IwAR3R0oInXiENBS5YSm6DklSXG_d-ehPNDZdSzGusA9nD_0L_GZ0xq-iQ3Mw


## License
This project is licensed under the MIT License - see the LICENSE file for details.