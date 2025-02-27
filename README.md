## Sentiment Analysis with BERT

Overview

This project implements a sentiment analysis model using BERT to classify text into three categories: positive, negative, and neutral. The dataset is preprocessed, trained on a transformer-based model, and evaluated using classification metrics.

## Features

-Parses custom text formats into structured CSV data

-Cleans and preprocesses text data

-Visualizes class distribution and text length

-Implements a BERT-based text classification model using simpletransformers

-Evaluates the model with classification reports and confusion matrices

-Provides a function for predicting sentiment with confidence scores and explanations



## Data Preprocessing

-Loads dataset and removes duplicates

-Visualizes class distribution using Seaborn

-Analyzes text length distribution

-Maps sentiment labels to numerical values (0: positive, 1: negative, 2: neutral)

## Model Training

-Model: bert-base-cased

-Training: Uses simpletransformers for training

Hyperparameters:

-num_train_epochs: 2

-learning_rate: 3e-5

-max_seq_length: 128

-train_batch_size: 8

-eval_batch_size: 16

## Model Evaluation

-Generates a classification report with precision, recall, and F1-score

-Plots a confusion matrix to analyze predictions
