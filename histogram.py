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


data = parse_args()

for col, series in data.select_dtypes(include=[np.float64]).items():
    print(col)
    plt.figure(figsize=(10, 6))
    data.groupby("Hogwarts House")[col].plot(kind="hist", alpha=0.5, bins=10)
    plt.title(col)
    plt.xlabel(f"{col} score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
