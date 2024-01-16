#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
from utils import HOUSE_COLORS, parse_args

data = parse_args("Show a pair plot of Hogwarts classes")
plt.figure(figsize=(20, 15))
sns.pairplot(data, hue="Hogwarts House", palette=HOUSE_COLORS)
plt.get_current_fig_manager().set_window_title("Pair Plot of Hogwarts Classes")
plt.show()
