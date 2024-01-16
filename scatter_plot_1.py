#!/usr/bin/env python

# TODO: clean the scatter plots
# TODO: scatter_plot_1v1

import matplotlib.pyplot as plt
import numpy as np
from parse_args import parse_args

data = parse_args("Show scatter plots of class grades by Hogwarts house.")
numeric_df = data.select_dtypes(include=[float])
log_transformed_df = numeric_df.apply(lambda x: np.log(x + 1 - x.min()))
log_means = log_transformed_df.mean()
log_std_devs = log_transformed_df.std()

plt.figure(figsize=(10, 6))
plt.scatter(log_means, log_std_devs)
plt.xlabel("Log Mean")
plt.ylabel("Log Standard Deviation")
plt.title("Scatter Plot of Log-Transformed Features")
for i, txt in enumerate(log_transformed_df.columns):
    plt.annotate(txt, (log_means.iloc[i], log_std_devs.iloc[i]))
plt.show()
