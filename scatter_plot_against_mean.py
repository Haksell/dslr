#!/usr/bin/python

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import HOUSE_COLORS, parse_args

_, WIDTH = DIMENSIONS = (3, 5)
TITLE = "Scatter Plots of Class Grades by Hogwarts House"

data = parse_args("Show scatter plots of class grades by Hogwarts house.")
numeric_columns = data.select_dtypes(include=[float]).columns
standardized_data = pd.DataFrame(
    MinMaxScaler().fit_transform(data[numeric_columns]), columns=numeric_columns
)
data["Average"] = standardized_data.mean(axis=1)
grouped_data = data.groupby("Hogwarts House")
fig = plt.figure(figsize=(20, 10))

for i, col in enumerate(numeric_columns):
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
