#!/usr/bin/env python3
"""Expectation Maximization"""


import numpy as np


def maximization(X, g):
    """Function that calculates the maximization step in the EM
    algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors
        for each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    cluster = np.sum(g, axis=0)
    cluster = np.sum(cluster)
    if int(cluster) != X.shape[0]:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]

    responsabilities = np.sum(g, axis=1)

    pi = responsabilities / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.matmul(g[i], X) / responsabilities[i]
        x_meaned = X - m[i]
        S[i] = np.matmul(g[i] * x_meaned.T, x_meaned) / responsabilities[i]

    return pi, m, S
