#!/usr/bin/env python3
"""Normalize constant"""


import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization (standardization)
    constants of a matrix:

    X: is the numpy.ndarray of shape (m, nx) to normalize
        m is the number of data points
        nx is the number of features
    Returns: the mean and standard deviation of each feature, respectively"""
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
