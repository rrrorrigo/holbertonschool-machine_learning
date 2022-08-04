#!/usr/bin/env python3
"""Expectation Maximization"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation step in the EM algorithm
    for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        l is the total log likelihood"""
    n, d = X.shape
    k = pi.shape[0]
    responsabilities_iter = np.zeros((k, n))

    for i in range(k):
        responsabilities_iter[i] = pi[i] * pdf(X, m[i], S[i])

    sum_responsabilities = np.sum(responsabilities_iter, axis=0)
    responsabilities = responsabilities_iter / sum_responsabilities

    log_likelihood = np.sum(np.log(sum_responsabilities))

    return responsabilities, log_likelihood
