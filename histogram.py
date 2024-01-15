#!/usr/bin/env python

import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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


_, WIDTH = DIMENSIONS = (3, 5)
BINS = 12

data = parse_args()
grouped_data = data.groupby("Hogwarts House")
numeric_columns = data.select_dtypes(include=[np.float64])
fig = plt.figure(figsize=(20, 10))
for i, (col, series) in enumerate(numeric_columns.items()):
    ax = plt.subplot2grid(DIMENSIONS, divmod(i, WIDTH))
    grouped_data[col].plot(kind="hist", alpha=0.5, bins=BINS, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    if i == 0:
        fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.1))
plt.tight_layout()
plt.show()
