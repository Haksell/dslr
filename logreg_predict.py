#!/usr/bin/python

import json
from logreg_train import predict_ovr, process_data
from numbers import Number
import numpy as np
from utils import parse_args
import sys

FILENAME_PREDICTIONS = "houses.csv"
EXPECTED_FEATURES = 16


def parse_parameters(parameters_filename):
    def check_array(array, name):
        assert (
            isinstance(array, list)
            and len(array) == EXPECTED_FEATURES
            and all(isinstance(x, Number) for x in means)
        ), f'"{name}" should be an array of {EXPECTED_FEATURES} numbers'

    try:
        parameters = json.load(open(parameters_filename))
        assert isinstance(parameters, dict), "JSON file should represent a dictionary"
        missing_keys = {"houses", "theta", "means", "stds"} - set(parameters.keys())
        assert (
            not missing_keys
        ), f"key{'s' if len(missing_keys)>=2 else ''} {missing_keys} not found"
        houses, means, stds, theta = (
            parameters["houses"],
            parameters["means"],
            parameters["stds"],
            parameters["theta"],
        )
        assert (
            isinstance(houses, dict)
            and len(houses) >= 2
            and all(
                isinstance(k, str) and isinstance(v, int) for k, v in houses.items()
            )
            and sorted(houses.values()) == list(range(len(houses)))
        ), '"houses" should be a correct mapping from house names to indices'
        check_array(means, "means")
        check_array(stds, "stds")
        assert (
            isinstance(theta, list)
            and len(theta) == EXPECTED_FEATURES + 1
            and all(isinstance(row, list) and len(row) == len(houses) for row in theta)
            and all(isinstance(x, Number) for row in theta for x in row)
        ), f'"theta" should be a {EXPECTED_FEATURES+1}x{len(houses)} matrix of numbers'
        return houses, np.asarray(means), np.asarray(stds), np.asarray(theta)
    except Exception as e:
        print(f"Failed to read parameters file: {e}")
        sys.exit(1)


def main():
    data, args = parse_args(
        "Train a Logistic Regression model using Gradient Descent.",
        additional_arguments=["parameters"],
    )
    houses, means, stds, theta = parse_parameters(args.parameters)
    X, y, *_ = process_data(
        data,
        houses=houses,
        means=means,
        stds=stds,
    )
    num_to_house = {v: k for k, v in houses.items()}
    predictions = predict_ovr(X, theta)
    output = "Index,Hogwarts House\n" + "".join(
        f"{i},{num_to_house[prediction]}\n" for i, prediction in enumerate(predictions)
    )
    try:
        open(FILENAME_PREDICTIONS, "w").write(output)
        print(f"Predictions saved to {FILENAME_PREDICTIONS}")
    except Exception as e:
        print(f'Failed to write predictions to file "{FILENAME_PREDICTIONS}": {e}')
        print(output)
    if np.isnan(y).sum() == 0:
        print(f"Accuracy: {(y == predictions).mean()}")


if __name__ == "__main__":
    main()
