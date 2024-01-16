#!/usr/bin/python

from math import exp, log as ln
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from utils import parse_args


def sigmoid(x):
    return exp(x) / (exp(x) + 1)


def logit(x):
    return -ln(1 / x - 1)


data = parse_args("Train a Logistic Regression model using Gradient Descent.")
X = data.iloc[:, 5:]
X = SimpleImputer(strategy="mean").fit_transform(X)
y = data["Hogwarts House"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")
