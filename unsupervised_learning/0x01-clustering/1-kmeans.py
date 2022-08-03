#!/usr/bin/env python3
"""Initialize cluster centroids"""


from mimetypes import init
import numpy as np
from sklearn import cluster

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
    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    n, d = X.shape
    return np.random.uniform(np.min(X, 0), np.max(X, 0), (k, d))


def kmeans(X, k, iterations=1000):
    """Function that performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to"""
    clusters = initialize(X, k)
    n, d = X.shape
    for i in range(iterations):
        clusters_copy = clusters.copy()
        # calculate distance with Euclidean Distance √np.sum(((x1-y1)², axis=2)
        x = np.sqrt(np.sum((X - clusters[:, np.newaxis]) ** 2, axis=2))
        # find the closest index daa point to cluster centroid
        closest_i = np.argmin(x, axis=0)
        for j in range(k):
            if len(X[closest_i == j]) == 0:
                clusters[j] = initialize(X, 1)
            else:
                clusters[j] = X[closest_i == j].mean(axis=0)
        if np.all(clusters_copy == clusters):
            break
    return clusters, closest_i
