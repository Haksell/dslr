from math import isnan
from describe import ft_mean, ft_skewness, ft_stdev, ft_variance
import numpy as np
import scipy.stats

datasets = [[round(x, 2) for x in np.random.normal(0, 1, i)] for i in range(20)]


def is_close(x, y):
    return isnan(x) and isnan(y) or abs(x - y) < 1e-7


def check_func(ft, lib):
    for dataset in datasets:
        assert is_close(ft(dataset), lib(dataset)), dataset


def test_mean():
    check_func(ft_mean, np.mean)


def test_var():
    check_func(lambda a: ft_variance(a, population=True), lambda a: np.var(a, ddof=0))
    check_func(lambda a: ft_variance(a, population=False), lambda a: np.var(a, ddof=1))


def test_stdev():
    check_func(lambda a: ft_stdev(a, population=True), lambda a: np.std(a, ddof=0))
    check_func(lambda a: ft_stdev(a, population=False), lambda a: np.std(a, ddof=1))


def test_skewness():
    check_func(
        lambda a: ft_skewness(a, population=True),
        lambda a: scipy.stats.skew(a, bias=True),
    )
    check_func(
        lambda a: ft_skewness(a, population=False),
        lambda a: scipy.stats.skew(a, bias=False),
    )


# def test_kurtosis():
#     pass
