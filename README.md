# Web-Mining

Project Overview:
Web-Mining is a Python-based project designed for the classification of newsgroup data using various machine learning models. The project utilizes Scikit-learn for model training and evaluation, and Streamlit for creating an interactive web app to demonstrate the classification process.

Features:
1) Classification of newsgroup data into predefined categories.
2) Utilizes multiple machine learning models from Scikit-learn.
3) Streamlit web app for easy interaction and visualization.
4) Custom data preprocessing and model evaluation functions.

Installation:
To get started with Web-Mining, clone this repository and install the required dependencies.

git clone https://github.com/Ashshashank/Web-Mining.git
cd Web-Mining
pip install -r requirements.txt

Usage:
The project is divided into three main Python scripts:

main.py - This script contains the core functionality for training and testing the machine learning models.
project_functions.py - Contains auxiliary functions for data preprocessing, model training, and testing.
app.py - Streamlit-based web application to demonstrate the project's capabilities interactively.

To run the Streamlit app:
streamlit run app.py

Models Used:
Multilayer Perceptron Classifier
Gaussian Naive Bayes
Bernoulli Naive Bayes
Multinomial Naive Bayes
K-Nearest Neighbors
Random Forest Classifier
Logistic Regression
Support Vector Machine
XGBoost

Data:
The project uses newsgroup data for classification, which is divided into two categories: sci.space and rec.autos. The data is preprocessed and vectorized before being fed into the models.

Contributing
Contributions to Web-Mining are welcome. Please ensure to update tests as appropriate.
