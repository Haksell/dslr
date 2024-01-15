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


WIDTH = 5
HEIGHT = 3

data = parse_args()
grouped_data = data.groupby("Hogwarts House")
numeric_columns = data.select_dtypes(include=[np.float64])
for i, (col, series) in enumerate(numeric_columns.items()):
    ax = plt.subplot2grid((HEIGHT, WIDTH), divmod(i, WIDTH))
    grouped_data[col].plot(kind="hist", alpha=0.5, bins=15, ax=ax)
    ax.set_title(col)
    ax.set_xlabel(f"{col} score")
    ax.set_ylabel("Frequency")
    ax.legend()


plt.tight_layout()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
