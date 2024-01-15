#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from parse_args import parse_args
from sklearn.preprocessing import MinMaxScaler


def get_numeric_columns(data):
    return data.select_dtypes(include=[np.float64])


def standardize_columns(data, columns):
    return pd.DataFrame(MinMaxScaler().fit_transform(data[columns]), columns=columns)


_, WIDTH = DIMENSIONS = (3, 5)
BINS = 12
TITLE = "Histograms of Class Grades by Hogwarts House"

data = parse_args("Show histograms of class grades by Hogwarts house.")
standardized_data = standardize_columns(data, get_numeric_columns(data).columns)
data["Average"] = standardized_data.mean(axis=1)
grouped_data = data.groupby("Hogwarts House")
fig = plt.figure(figsize=(20, 10))
for i, col in enumerate(get_numeric_columns(data)):
    ax = plt.subplot2grid(DIMENSIONS, divmod(i, WIDTH))
    grouped_data[col].plot(kind="hist", alpha=0.5, bins=BINS, ax=ax)
    ax.set_xlabel(col)
    ax.set_ylabel("")
    if i == 0:
        fig.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15))
fig.suptitle(TITLE, fontsize=14)
fig.canvas.manager.set_window_title(TITLE)
plt.tight_layout()
plt.show()
