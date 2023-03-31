# -*- coding: utf-8 -*-
"""breast_cancer

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AH2_JlsrhVRMQsn9WdHiCTI1PpmbITpK

### Logistic Regression - Breast Cancer prediction.

### Import the libraries
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

"""### Import the dataset"""

cancer_dataset = pd.read_csv('breast_cancer.csv')
x = cancer_dataset.iloc[:,1:-1]
y = cancer_dataset.iloc[:,-1]

"""Checking for NAN values."""

cancer_dataset.isna().sum()

"""Checking for null values."""

cancer_dataset.isnull().sum()

"""### Splitting the Dataset into training and testing parts."""

trainx, testx, trainy, testy = train_test_split(x,y,test_size=0.2, random_state=42)
print(len(testy))

"""Feature Selection using RFE"""

logreg = LogisticRegression(max_iter=1000)
rfe = RFE(logreg, n_features_to_select=10)
trainx = rfe.fit_transform(trainx, trainy)
testx = rfe.transform(testx)

"""### Training the Machine Learning model on the Training data portion."""

logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
logreg.fit(trainx, trainy)

"""### Predtion of results."""

y_hat = logreg.predict(testx)

"""### Evaluating the model using different metrics."""

#confusion matrix
cm = confusion_matrix(testy, y_hat)
print("Confusion Matrix: \n",cm)
acc_score = accuracy_score(testy,y_hat)
print("Accuracy score:",acc_score)

# k-fold cross validation.
acc = cross_val_score(estimator=logreg, X=trainx, y=trainy, cv =10)
print("Accuracy: {:.2f}%".format(acc.mean()*100))
print("Standard Deviation: {:.2f}%".format(acc.std()*100))