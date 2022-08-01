#!/usr/bin/env python3
"""Initialize cluster centroids"""


import numpy as np


def initialize(X, k):
    """Function that initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset that
    will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate
    uniform distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
        values of X along each dimension in d
        The maximum values for the distribution should be the maximum
        values of X along each dimension in d

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure"""
    n, d = X.shape
    try:
        return np.random.uniform(np.min(X, 0), np.max(X, 0), (k, d))
    except:
        return None
