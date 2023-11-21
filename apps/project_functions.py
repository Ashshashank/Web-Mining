#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:49:25 2023

@author: shashank
"""

import os
import chardet
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('/Users/shashank/Documents/WebMining/HW-4-NewsG-Classification-Py')
from main import model_train, model_test
import xgboost as xgb




"""def read_text_file(path):
    with open(path, 'r') as file:
        content = file.read().replace('\n', '')
    return content"""
def read_text_file(path):
    with open(path, 'rb') as file:  # Open the file in binary mode
        raw_data = file.read()
        encoding = chardet.detect(raw_data)['encoding']  # Detect encoding
        if encoding is not None:
            return raw_data.decode(encoding).replace('\n', '')
        else:
            return raw_data.decode('utf-8', errors='replace').replace('\n', '')  # Fallback to utf-8


def get_model(model_name):
    if model_name == "model_nn":
        return MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500, random_state=1, solver='adam', activation='relu')
    elif model_name == "model_nb":
        return GaussianNB()
    elif model_name == "model_nbb":
        return BernoulliNB()
    elif model_name == "model_mnb":
        return MultinomialNB()
    elif model_name == "model_knn":
        return KNeighborsClassifier(n_neighbors=2)
    elif model_name == "model_rf":
        return RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_name == "model_lr":
        return LogisticRegression(solver='liblinear')
    elif model_name == "model_svm":
        return SVC(kernel='rbf', gamma='auto', probability=True)
    elif model_name == "model_svm1":
        return SVC(kernel='rbf', random_state=1, gamma=0.001, C=20)
    elif model_name == "model_xgb":
        return xgb.XGBClassifier(colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_depth=7, reg_alpha=0, reg_lambda=1, subsample=0.8)
    else:
        return None

def calculate_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train.toarray(), y_train)
    predictions = model.predict(X_test.toarray())
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    cm = confusion_matrix(y_test, predictions)
    return accuracy, precision, recall, f1, cm

def get_model_metrics(model_name, data, tags):
    vectorizer = TfidfVectorizer() if "tfidf" in model_name else CountVectorizer()
    X = vectorizer.fit_transform(data)
    y = pd.Series(tags)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = get_model(model_name)
    
    if model is not None:
        return calculate_metrics(model, X_train, X_test, y_train, y_test)
    else:
        return "Model not found"

def load_data(path):
    return pd.read_csv(path)





def load_and_preprocess_data():
    # Paths to your data
    newsgroup1 = "sci.space"
    newsgroup2 = "rec.autos"
    path_to_20Newsgroups = "/Users/shashank/Documents/WebMining/HW-4-NewsG-Classification-Py/data/20Newsgroups"
    train_folder = os.path.join(path_to_20Newsgroups, "20news-bydate-train")
    test_folder = os.path.join(path_to_20Newsgroups, "20news-bydate-test")

    # Load data
    doc1_train_path = os.path.join(train_folder, newsgroup1)
    doc1_test_path = os.path.join(test_folder, newsgroup1)
    doc2_train_path = os.path.join(train_folder, newsgroup2)
    doc2_test_path = os.path.join(test_folder, newsgroup2)

    count = 100  # Limit the number of documents

    # Read and store data
    doc1_train, doc1_test, doc2_train, doc2_test = [], [], [], []
    for i in range(count):
        doc1_train.append(read_text_file(os.path.join(doc1_train_path, os.listdir(doc1_train_path)[i])))
        doc1_test.append(read_text_file(os.path.join(doc1_test_path, os.listdir(doc1_test_path)[i])))
        doc2_train.append(read_text_file(os.path.join(doc2_train_path, os.listdir(doc2_train_path)[i])))
        doc2_test.append(read_text_file(os.path.join(doc2_test_path, os.listdir(doc2_test_path)[i])))

    # Combine data
    train_data = doc1_train + doc2_train
    test_data = doc1_test + doc2_test

    # Labels
    train_tags = ['Positive' for _ in doc1_train] + ['Negative' for _ in doc2_train]
    test_tags = ['Positive' for _ in doc1_test] + ['Negative' for _ in doc2_test]

    return train_data, test_data, train_tags, test_tags

# Function to run a specified model
def run_model(model_name):
    # Load and preprocess data
    train_data, test_data, train_tags, test_tags = load_and_preprocess_data()

    # Define the vectorizer based on the model's requirements
    vectorizer = CountVectorizer() if model_name in ['model_nn', 'model_nb'] else TfidfVectorizer()

    # Transform your data
    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)

    # Train the model
    model = model_train(x_train, train_tags, model_name, models_path="")

    # Test the model and get the results
    predicted_test, accuracy_test, precision_test, recall_test, f1_score_value, cm_test = model_test(model, x_test, test_tags)
    
    # Ensure that all metrics are available before creating the DataFrame
    if all(v is not None for v in [accuracy_test, precision_test, recall_test, f1_score_value, cm_test]):
        # Format the results into a DataFrame
        results = pd.DataFrame({
            'Model Name': [model_name],
            'Accuracy': [accuracy_test],
            'Precision': [precision_test],
            'Recall': [recall_test],
            'F1 Score': [f1_score_value],
            'Confusion Matrix': [cm_test.to_numpy().tolist()]  # Optionally format the confusion matrix
        })
    else:
        # If any of the metrics are missing, return an appropriate message or empty DataFrame
        print(f"Metrics not found for model: {model_name}")
        results = pd.DataFrame()
    
    
    # Format the results into a DataFrame
    results = pd.DataFrame({
        'Model Name': [model_name],
        'Accuracy': [accuracy_test],
        'Precision': [precision_test],
        'Recall': [recall_test],
        'F1 Score': [f1_score_value],
        'Confusion Matrix': [cm_test.to_numpy().tolist()]  # Optionally format the confusion matrix
    })

    return results
