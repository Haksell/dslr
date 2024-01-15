#!/usr/bin/env python

import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Describe numeric data of the given dataset."
    )
    parser.add_argument("filename", type=str, help="Filename of the CSV.")
    args = parser.parse_args()
    try:
        return pd.read_csv(args.filename, index_col="Index")
    except FileNotFoundError:
        print(f"Error: File {args.filename} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_numeric_columns(data):
    return data.select_dtypes(include=[np.float64])


def normalize_columns(data, columns):
    return pd.DataFrame(StandardScaler().fit_transform(data[columns]), columns=columns)


_, WIDTH = DIMENSIONS = (3, 5)
BINS = 12
TITLE = "Histograms of Class Grades by Hogwarts House"

data = parse_args()
grouped_data = data.groupby("Hogwarts House")
normalized_data = normalize_columns(data, get_numeric_columns(data).columns)
data["Average"] = normalized_data.mean(axis=1)
fig = plt.figure(figsize=(20, 10))
for i, (col, series) in enumerate(get_numeric_columns(data).items()):
    ax = plt.subplot2grid(DIMENSIONS, divmod(i, WIDTH))
    grouped_data[col].plot(kind="hist", alpha=0.5, bins=BINS, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("")
    if i == 0:
        fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
fig.suptitle(TITLE, fontsize=16)
fig.canvas.manager.set_window_title(TITLE)
plt.tight_layout()
plt.show()
