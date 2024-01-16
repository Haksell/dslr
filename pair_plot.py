#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
from utils import HOUSE_COLORS, parse_args


def abbreviate_column_names(data):
    def abbreviate_column_name(column):
        return "".join(w[0] for w in column.split()) if " " in column else column[:7]

    columns = list(data.select_dtypes(include=[float]).columns)
    abbreviations = {c: abbreviate_column_name(c) for c in columns}
    return data.rename(columns=abbreviations)


data = parse_args("Show a pair plot of Hogwarts classes")
data = abbreviate_column_names(data)

pair_plot = sns.pairplot(
    data, hue="Hogwarts House", palette=HOUSE_COLORS, height=2, aspect=1.5
)
pair_plot._legend.remove()

for ax in pair_plot.axes.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

plt.get_current_fig_manager().set_window_title("Pair Plot of Hogwarts Classes")
plt.show()
