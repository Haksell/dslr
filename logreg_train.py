#!/usr/bin/python

import json
import numpy as np
from sklearn.model_selection import KFold
from utils import parse_args

FILENAME_PARAMETERS = "parameters.json"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, *, learning_rate=0.01, num_iters=300):
    theta = np.zeros(X.shape[1])
    for _ in range(num_iters):
        y_hat = sigmoid(X @ theta)
        gradient = X.T @ (y_hat - y) / len(y)
        theta -= learning_rate * gradient
    return theta


def train_ovr(X, y, num_labels):
    return np.column_stack(
        [gradient_descent(X, np.where(y == i, 1, 0)) for i in range(num_labels)]
    )


def predict_ovr(X, all_theta):
    return np.argmax(X @ all_theta, axis=1)


def process_data(data, *, houses=None, means=None, stds=None):
    data["Best Hand"] = data["Best Hand"].map({"Left": 0, "Right": 1})
    data["Month"] = data["Birthday"].apply(lambda x: int(x[5:7]))
    data["Day"] = data["Birthday"].apply(lambda x: int(x[8:10]))
    X = data.iloc[:, 4:].values
    if means is None:
        means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), means, X)
    if stds is None:
        stds = np.std(X, axis=0)
    X = (X - means) / stds
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = data["Hogwarts House"]
    if houses is None:
        houses = {house: i for i, house in enumerate(y.unique())}
    y = data["Hogwarts House"].map(houses)
    return X, y, houses, means, stds


def main():
    data, args = parse_args(
        "Train a Logistic Regression model using Gradient Descent.", flags=["--debug"]
    )
    X, y, houses, means, stds = process_data(data)
    if args.debug:
        accuracies = []
        for fold_idx, (train_index, test_index) in enumerate(
            KFold(n_splits=5, shuffle=True).split(X)
        ):
            X_train, X_test, y_train, y_test = (
                X[train_index],
                X[test_index],
                y[train_index],
                y[test_index],
            )
            theta = train_ovr(X_train, y_train, len(houses))
            predictions = predict_ovr(X_test, theta)
            accuracy = (y_test == predictions).mean()
            accuracies.append(accuracy)
            print(f"Accuracy for fold {fold_idx}: {accuracy:.3f}")
        print(f"Mean accuracy: {np.mean(accuracies):.3f}")
    theta = train_ovr(X, y, len(houses))
    try:
        json.dump(
            {
                "houses": houses,
                "means": list(means),
                "stds": list(stds),
                "theta": list(map(list, theta)),
            },
            open(FILENAME_PARAMETERS, "w"),
        )
        print(f"Model parameters saved to {FILENAME_PARAMETERS}")
    except Exception as e:
        print(f'Failed to write arguments to file "{FILENAME_PARAMETERS}": {e}')
        print(theta)


if __name__ == "__main__":
    main()
