#!/usr/bin/python

import json
import sys
import numpy as np
from logreg_train import predict_ovr, process_data
from utils import parse_args

FILENAME_PREDICTIONS = "houses.csv"

data, args = parse_args(
    "Train a Logistic Regression model using Gradient Descent.",
    additional_arguments=["parameters"],
)
try:
    parameters = json.load(open(args.parameters))
except Exception as e:
    print(f"Failed to read parameters file: {e}")
    sys.exit(1)
X, y, *_ = process_data(
    data,
    means=parameters["means"],
    stds=parameters["stds"],
    houses=parameters["houses"],
)
theta = np.asarray(parameters["theta"])
num_to_house = {v: k for k, v in parameters["houses"].items()}
predictions = predict_ovr(X, theta)
output = "Index,Hogwarts House\n" + "".join(
    f"{i},{num_to_house[prediction]}\n" for i, prediction in enumerate(predictions[:10])
)
try:
    open(FILENAME_PREDICTIONS, "w").write(output)
    print(f"Predictions saved to {FILENAME_PREDICTIONS}")
except Exception as e:
    print(f'Failed to write predictions to file "{FILENAME_PREDICTIONS}": {e}')
    print(output)
if np.isnan(y).sum() == 0:
    print(f"Accuracy: {(y == predictions).mean()}")
