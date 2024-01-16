#!/usr/bin/python

import sys
from matplotlib import pyplot as plt
from utils import HOUSE_COLORS, parse_args


def simplify_string(s):
    return "".join(c.upper() for c in s if c.isalpha())


def check_class_exists(c, u, upper_to_original, errors):
    if u not in upper_to_original:
        errors.append(f'"{c}"')


data, args = parse_args(
    "Show scatter plots of class grades by Hogwarts house.",
    additional_arguments=["class1", "class2"],
)
class1, class2 = args.class1, args.class2
upper1, upper2 = map(simplify_string, [class1, class2])
if upper1 == upper2:
    print("Impossible to plot a class against itself")
    sys.exit(1)
upper_to_original = dict()
for column in data.columns:
    if data[column].dtype != "object":
        upper_column = simplify_string(column)
        upper_to_original[upper_column] = column
        data.rename(columns={column: upper_column}, inplace=True)

errors = []
check_class_exists(class1, upper1, upper_to_original, errors)
check_class_exists(class2, upper2, upper_to_original, errors)
if errors:
    print(f'Class{"es" if len(errors)==2 else ""} {" and ".join(errors)} not found.')
    print("Class must be one of:")
    for _, original in sorted(upper_to_original.items()):
        print(f"- {original}")
    sys.exit(1)

original1 = upper_to_original[upper1]
original2 = upper_to_original[upper2]

title = f"{original1} vs {original2}"
fig, ax = plt.subplots()
for house in HOUSE_COLORS:
    house_data = data[data["Hogwarts House"] == house]
    ax.scatter(
        house_data[upper1],
        house_data[upper2],
        color=HOUSE_COLORS[house],
        label=house,
        alpha=0.5,
    )

ax.set_xlabel(original1)
ax.set_ylabel(original2)
ax.set_title(title)
fig.canvas.manager.set_window_title(title)
ax.legend()
plt.show()
