#!/usr/bin/env python3
"""mean covariance"""


import numpy as np


def mean_cov(X):
    """Function that calculate the covariance of dataset

    X is a numpy.ndarray of shape (n, d) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
    If X is not a 2D numpy.ndarray, raise a TypeError with the message X
    must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the message X must contain
    multiple data points
    Returns: mean, cov:
        mean is a numpy.ndarray of shape (1, d) containing the mean of the
        data set
        cov is a numpy.ndarray of shape (d, d) containing the covariance matrix
        of the data set"""
    if len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    data, dim = X.shape
    if data < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.mean(X[:], axis=0).reshape(1, dim)
    x = X - mean
    cov = (x.T @ x) / (data - 1)
    return mean, cov
