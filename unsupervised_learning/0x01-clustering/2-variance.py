#!/usr/bin/env python3
"""variance intra cluster"""


import numpy as np


def variance(X, C): 
    """Function that calculates the total intra-cluster variance for
    a data set:

    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means for
    each cluster
    Returns: var, or None on failure
    var is the total variance"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    distance = np.sqrt(np.sum((X - C[:, np.newaxis]) ** 2, axis=2))
    var = np.sum(np.sum(distance.min(axis=0)**2))

    return var
