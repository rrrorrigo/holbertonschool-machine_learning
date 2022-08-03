#!/usr/bin/env python3
"""tests for the optimum number of clusters by variance"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """that tests for the optimum number of clusters by variance:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters
    to check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters
    to check for (inclusive)
    iterations is a positive integer containing the maximum number of
    iterations for K-means
    Returns: results, d_vars, or None, None on failure
        results is a list containing the outputs of K-means for each
        cluster size
        d_vars is a list containing the difference in variance from the
        smallest cluster size for each cluster size"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    results, d_vars = [], []
    for k in range(kmin, kmax + 1):
        C, closest_i = kmeans(X, k)
        results.append((C, closest_i))
        if k == kmin:
            first_variance = variance(X, C)
        var = variance(X, C)
        d_vars.append(first_variance - var)
    return results, d_vars
