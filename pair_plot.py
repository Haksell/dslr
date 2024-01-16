#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
from utils import HOUSE_COLORS, parse_args

data = parse_args("Show a pair plot of Hogwarts classes")
data.rename(
    columns={
        c: "".join(w[0] for w in c.split()) if " " in c else c[:7]
        for c in data.select_dtypes(include=[float]).columns
    },
    inplace=True,
)

pair_plot = sns.pairplot(
    data,
    hue="Hogwarts House",
    palette=HOUSE_COLORS,
    height=2,
    aspect=1.5,
    plot_kws={"s": 20},
)
pair_plot._legend.remove()

for ax in pair_plot.axes.flatten():
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

plt.get_current_fig_manager().set_window_title("Pair Plot of Hogwarts Classes")
plt.show()
