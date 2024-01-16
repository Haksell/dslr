#!/usr/bin/env python

from matplotlib import pyplot as plt
import pandas as pd
from parse_args import parse_args
from sklearn.preprocessing import MinMaxScaler


def get_numeric_columns(data):
    return data.select_dtypes(include=[float])


def standardize_columns(data, columns):
    return pd.DataFrame(MinMaxScaler().fit_transform(data[columns]), columns=columns)


_, WIDTH = DIMENSIONS = (3, 5)
TITLE = "Scatter Plots of Class Grades by Hogwarts House"
HOUSE_COLORS = {
    "Gryffindor": "#D62728",
    "Hufflepuff": "#FF7F0E",
    "Ravenclaw": "#1F77B4",
    "Slytherin": "#2CA02C",
}

data = parse_args("Show scatter plots of class grades by Hogwarts house.")
standardized_data = standardize_columns(data, get_numeric_columns(data).columns)
data["Average"] = standardized_data.mean(axis=1)
grouped_data = data.groupby("Hogwarts House")
fig = plt.figure(figsize=(20, 10))

for i, col in enumerate(get_numeric_columns(data)):
    if i == 13:
        break
    ax = plt.subplot2grid(DIMENSIONS, divmod(i, WIDTH))
    for name, group in grouped_data:
        ax.scatter(
            group[col],
            group["Average"],
            label=name,
            color=HOUSE_COLORS[name],
            alpha=0.5,
        )
    ax.set_xlabel(col)
    ax.set_ylabel("Average")
    if i == 0:
        fig.legend(loc="lower right", bbox_to_anchor=(0.74, 0.14))

fig.suptitle(TITLE, fontsize=14)
fig.canvas.manager.set_window_title(TITLE)
plt.tight_layout()
plt.show()
