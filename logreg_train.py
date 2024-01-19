#!/usr/bin/python

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from utils import parse_args


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, *, learning_rate=0.01, num_iters=300):
    theta = np.zeros(X.shape[1])
    for _ in range(num_iters):
        y_hat = sigmoid(X @ theta)
        gradient = X.T @ (y_hat - y) / len(y)
        theta -= learning_rate * gradient
    return theta


def predict_ovr(X, all_theta):
    return np.argmax(X @ all_theta, axis=1)


def split_data(data):
    data["Best Hand"] = data["Best Hand"].map({"Left": 0, "Right": 1})
    data["Month"] = data["Birthday"].apply(lambda x: int(x[5:7]))
    data["Day"] = data["Birthday"].apply(lambda x: int(x[8:10]))
    X = data.iloc[:, 4:].values
    means = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), means, X)
    X = (X - means) / np.std(X, axis=0)
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = data["Hogwarts House"]
    houses = {house: i for i, house in enumerate(y.unique())}
    y = data["Hogwarts House"].map(houses)
    num_labels = len(houses)
    return X, y, num_labels


def main():
    data, args = parse_args(
        "Train a Logistic Regression model using Gradient Descent.", flags=["--debug"]
    )
    X, y, num_labels = split_data(data)

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
            theta = np.column_stack(
                [
                    gradient_descent(X_train, np.where(y_train == i, 1, 0))
                    for i in range(num_labels)
                ]
            )
            predictions = predict_ovr(X_test, theta)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            print(f"Accuracy for fold {fold_idx}: {accuracy:.3f}")
        print(f"Mean accuracy: {np.mean(accuracies):.3f}")


if __name__ == "__main__":
    main()
