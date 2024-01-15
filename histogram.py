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
BINS = 16
TITLE = "Histograms of Class Grades by Hogwarts House"
# HOUSE_COLORS = {
#     "Gryffindor": "#AE0001",
#     "Hufflepuff": "#F0C75E",
#     "Ravenclaw": "#222F5B",
#     "Slytherin": "#2A623D",
# }

data = parse_args("Show histograms of class grades by Hogwarts house.")
standardized_data = standardize_columns(data, get_numeric_columns(data).columns)
data["Average"] = standardized_data.mean(axis=1)
grouped_data = data.groupby("Hogwarts House")
fig = plt.figure(figsize=(20, 10))
for i, col in enumerate(get_numeric_columns(data)):
    if i == 13:
        i = 14
    ax = plt.subplot2grid(DIMENSIONS, divmod(i, WIDTH))
    min_val = data[col].min()
    max_val = data[col].max()
    bins = np.linspace(min_val, max_val, BINS)
    for name, group in grouped_data:
        ax.hist(group[col], bins=bins, alpha=0.5, label=name)
    ax.set_xlabel(col)
    if i == 0:
        fig.legend(loc="lower right", bbox_to_anchor=(0.74, 0.14))
fig.suptitle(TITLE, fontsize=14)
fig.canvas.manager.set_window_title(TITLE)
plt.tight_layout()
plt.show()
