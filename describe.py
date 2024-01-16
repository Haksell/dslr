#!/usr/bin/env python

from math import ceil, floor, isnan, nan, sqrt
from utils import parse_args


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    return (
        quicksort([x for x in arr if x < pivot])
        + [x for x in arr if x == pivot]
        + quicksort([x for x in arr if x > pivot])
    )


def ft_sum(arr):
    res = 0
    for x in arr:
        res += x
    return res


def ft_mean(arr):
    return ft_sum(arr) / len(arr) if arr else nan


def ft_variance(arr, *, population=False):
    divisor = len(arr) if population else len(arr) - 1
    if divisor <= 0:
        return nan
    m = ft_mean(arr)
    return ft_sum((x - m) ** 2 for x in arr) / divisor


def ft_stdev(arr, *, population=False):
    return sqrt(ft_variance(arr, population=population))


def ft_skewness(data, *, population=False):
    n = len(data)
    if n <= 1 or all(x == data[0] for x in data):
        return nan
    if n == 2:
        return 0
    mean = ft_mean(data)
    var = ft_variance(data, population=True)
    skewness = ft_mean([(x - mean) ** 3 for x in data]) / var**1.5
    if population:
        return skewness
    else:
        return skewness * sqrt(n * (n - 1)) / (n - 2)


def ft_kurtosis(data, *, population=False):
    n = len(data)
    if n <= 1 or all(x == data[0] for x in data):
        return nan
    if n == 2:
        return -2
    if n == 3:
        return -1.5
    mean = sum(data) / n
    m2 = sum((item - mean) ** 2 for item in data) / n
    m4 = sum((item - mean) ** 4 for item in data) / n
    kurtosis = m4 / m2**2
    if population:
        return kurtosis - 3
    else:
        return ((n**2 - 1.0) * kurtosis - 3 * (n - 1) ** 2.0) / ((n - 2) * (n - 3))


def ft_min(arr):
    return arr[0] if arr else nan


def ft_max(arr):
    return arr[-1] if arr else nan


def ft_percentile(arr, p):
    if not arr:
        return nan
    idx = p / 100 * (len(arr) - 1)
    if idx.is_integer():
        return arr[int(idx)]
    before = arr[floor(idx)]
    after = arr[ceil(idx)]
    interp = idx % 1
    return before * (1 - interp) + after * interp


def ft_median(arr):
    return ft_percentile(arr, 50)


def ft_median_absolute_deviation(arr):
    if not arr:
        return nan
    median = ft_median(arr)
    return ft_median(quicksort([abs(x - median) for x in arr]))


def ft_range(arr):
    return ft_max(arr) - ft_min(arr)


def ft_interquartile_range(arr):
    return ft_percentile(arr, 75) - ft_percentile(arr, 25)


def print_row(strings, widths):
    print(
        "  ".join(
            s.ljust(w) if i == 0 else s.rjust(w)
            for i, (s, w) in enumerate(zip(strings, widths))
        )
    )


def format_float(x):
    return "NaN" if isnan(x) else f"{x:.6f}"


def main():
    data = parse_args("Describe numeric data of the given dataset.")
    # print(data.describe(), "\n\n")
    pairs = [
        ("count", len),
        ("mean", ft_mean),
        ("var", ft_variance),
        ("std", ft_stdev),
        ("mad", ft_median_absolute_deviation),
        ("range", ft_range),
        ("iqr", ft_interquartile_range),
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
        values = quicksort([x for x in series if not isnan(x)])
        for line, func in zip(lines, functions):
            line.append(format_float(func(values)))

    widths = [max(map(len, strings)) for strings in zip(columns, *lines)]
    print_row(columns, widths)
    for line in lines:
        print_row(line, widths)


if __name__ == "__main__":
    main()
