#!/usr/bin/env python

from math import ceil, floor, isnan, nan, sqrt
import numpy as np

from parse_args import parse_args


def ft_sum(arr):
    res = 0
    for x in arr:
        res += x
    return res


def ft_mean(arr):
    return ft_sum(arr) / len(arr) if arr else nan


def ft_stdev(arr):
    if len(arr) <= 1:
        return nan
    m = ft_mean(arr)
    return sqrt(ft_sum((x - m) ** 2 for x in arr) / (len(arr) - 1))


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


def percentile(arr, p):
    assert 0 <= p <= 100
    if p == 0:
        return min(arr)
    if p == 100:
        return max(arr)
    if not arr:
        return nan
    arr = sorted(arr)
    idx = p / 100 * (len(arr) - 1)
    before = arr[floor(idx)]
    after = arr[ceil(idx)]
    interp = idx % 1
    return before * (1 - interp) + after * interp


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
# print(data.describe(), end="\n\n")

columns = [""]
counts = ["count"]
means = ["mean"]
stds = ["std"]
mins = ["min"]
q1 = ["25%"]
q2 = ["50%"]
q3 = ["75%"]
maxs = ["max"]

for column, series in data.select_dtypes(include=[np.float64]).items():
    columns.append(column)
    values = [x for x in series if not isnan(x)]
    counts.append(format_float(len(values)))
    means.append(format_float(ft_mean(values)))
    stds.append(format_float(ft_stdev(values)))
    mins.append(format_float(ft_min(values)))
    q1.append(format_float(percentile(values, 25)))
    q2.append(format_float(percentile(values, 50)))
    q3.append(format_float(percentile(values, 75)))
    maxs.append(format_float(ft_max(values)))

widths = [
    max(map(len, strings))
    for strings in zip(columns, counts, means, stds, mins, q1, q2, q3, maxs)
]

print_row(columns, widths)
print_row(counts, widths)
print_row(means, widths)
print_row(stds, widths)
print_row(mins, widths)
print_row(q1, widths)
print_row(q2, widths)
print_row(q3, widths)
print_row(maxs, widths)
