#!/usr/bin/env python

from math import ceil, floor, isnan, nan, sqrt
from utils import parse_args


def ft_sum(arr):
    res = 0
    for x in arr:
        res += x
    return res


def ft_mean(arr):
    return ft_sum(arr) / len(arr) if arr else nan


def ft_var(arr):
    if len(arr) <= 1:
        return nan
    m = ft_mean(arr)
    return ft_sum((x - m) ** 2 for x in arr) / (len(arr) - 1)


def ft_stdev(arr):
    return sqrt(ft_var(arr))


def ft_min(arr):
    if not arr:
        return nan
    res = arr[0]
    for x in arr[1:]:
        if x < res:
            res = x
    return res


def ft_max(arr):
    if not arr:
        return nan
    res = arr[0]
    for x in arr[1:]:
        if x > res:
            res = x
    return res


def ft_percentile(arr, p):
    # Assumes arr is already sorted
    assert 0 <= p <= 100
    if p == 0:
        return min(arr)
    if p == 100:
        return max(arr)
    if not arr:
        return nan
    idx = p / 100 * (len(arr) - 1)
    before = arr[floor(idx)]
    after = arr[ceil(idx)]
    interp = idx % 1
    return before * (1 - interp) + after * interp


def ft_median(arr):
    # Assumes arr is already sorted
    return ft_percentile(arr, 50)


def ft_median_absolute_deviation(arr):
    if not arr:
        return nan
    median = ft_median(arr)
    return ft_median(sorted([abs(x - median) for x in arr]))


def print_row(strings, widths):
    print(
        "  ".join(
            s.ljust(w) if i == 0 else s.rjust(w)
            for i, (s, w) in enumerate(zip(strings, widths))
        )
    )


def format_float(x):
    return "NaN" if isnan(x) else f"{x:.6f}"


data = parse_args("Describe numeric data of the given dataset.")
# print(data.describe(), "\n\n")
pairs = [
    ("count", len),
    ("mean", ft_mean),
    ("var", ft_var),
    ("std", ft_stdev),
    ("mad", ft_median_absolute_deviation),
    ("min", ft_min),
    ("25%", lambda a: ft_percentile(a, 25)),
    ("50%", ft_median),
    ("75%", lambda a: ft_percentile(a, 75)),
    ("max", ft_max),
]
lines = [[name] for name, _ in pairs]
functions = [func for _, func in pairs]
columns = [""]

for column, series in data.select_dtypes(include=[float]).items():
    columns.append(column)
    values = sorted(x for x in series if not isnan(x))
    for line, func in zip(lines, functions):
        line.append(format_float(func(values)))

widths = [max(map(len, strings)) for strings in zip(columns, *lines)]
print_row(columns, widths)
for line in lines:
    print_row(line, widths)
