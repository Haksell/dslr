#!/usr/bin/python

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from utils import parse_args


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(X, y, learning_rate, num_iters):
    theta = np.zeros(X.shape[1])
    for _ in range(num_iters):
        gradient = X.T @ (sigmoid(X @ theta) - y) / len(y)
        theta -= learning_rate * gradient
    return theta


def predict(X, all_theta):
    return np.argmax(X @ all_theta, axis=1)


def main():
    data, args = parse_args(
        "Train a Logistic Regression model using Gradient Descent.", flags=["--debug"]
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(
        SimpleImputer(strategy="mean").fit_transform(data.iloc[:, 5:])
    )
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    y = data["Hogwarts House"].apply(
        {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}.get
    )
    num_labels = len(np.unique(y))
    _, n = X.shape

    learning_rate = 0.01
    num_iters = 300

    if args.debug:
        kf = KFold(n_splits=5, shuffle=True)
        accuracies = []
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test, y_train, y_test = (
                X[train_index],
                X[test_index],
                y[train_index],
                y[test_index],
            )
            all_theta = np.zeros((n, num_labels))
            for i in range(num_labels):
                temp_y = np.where(y_train == i, 1, 0)
                all_theta[:, i] = gradient_descent(
                    X_train, temp_y, learning_rate, num_iters
                )
            predictions = predict(X_test, all_theta)
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            print(f"Accuracy for fold {fold_idx}: {accuracy:.3f}")

        print(f"Mean accuracy: {np.mean(accuracies):.3f}")


if __name__ == "__main__":
    main()
