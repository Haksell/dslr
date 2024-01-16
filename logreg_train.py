#!/usr/bin/python

from math import exp, log as ln
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from utils import parse_args


def sigmoid(x):
    return exp(x) / (exp(x) + 1)


def logit(x):
    return -ln(1 / x - 1)


def main():
    data, args = parse_args(
        "Train a Logistic Regression model using Gradient Descent.", flags=["--debug"]
    )
    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(data.iloc[:, 5:])
    )
    y = data["Hogwarts House"]
    model = OneVsRestClassifier(LogisticRegression())
    if args.debug:
        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=KFold(n_splits=5, shuffle=True),
            scoring="accuracy",
        )
        print(f"Cross-Validation Accuracy Scores: {[round(x,3) for x in cv_scores]}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.3f}")
    model.fit(X, y)


if __name__ == "__main__":
    main()