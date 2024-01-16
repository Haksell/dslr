#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from utils import parse_args

TITLE = "Scatter Plot of Log-Transformed Features"

data = parse_args("Show a scatter plot of classes means and standard deviations.")
numeric_df = data.select_dtypes(include=[float])
log_transformed_df = numeric_df.apply(lambda x: np.log(x + 1 - x.min()))
log_means = log_transformed_df.mean()
log_std_devs = log_transformed_df.std()

plt.figure(figsize=(10, 6))
plt.scatter(log_means, log_std_devs)
plt.xlabel("Log Mean")
plt.ylabel("Log Standard Deviation")
plt.title(TITLE)
plt.get_current_fig_manager().set_window_title(TITLE)
for i, txt in enumerate(log_transformed_df.columns):
    plt.annotate(txt, (log_means.iloc[i], log_std_devs.iloc[i]))
plt.show()
