#!/usr/bin/env python3
"""Expectation Maximization"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """that finds the best number of clusters for a GMM using the Bayesian
    Information Criterion:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to
    check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters to
    check for (inclusive)
    If kmax is None, kmax should be set to the maximum number of
    clusters possible
    iterations is a positive integer containing the maximum number of
    iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM algorithm
    verbose is a boolean that determines if the EM algorithm should print
    information to the standard output

    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
        pi is a numpy.ndarray of shape (k,) containing the cluster priors for
        the best number of clusters
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
        the best number of clusters
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
        value for each cluster size tested"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] <= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    all_pi = []
    all_m = []
    all_S = []
    all_lh = []
    all_b = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        all_pi.append(pi)
        all_m.append(m)
        all_S.append(S)
        all_lh.append(log_likelihood)

        b = (k * d * (d + 1) / 2) + (d * k) + (k - 1)
        bic = b * np.log(n) - 2 * log_likelihood
        all_b.append(bic)

    all_lh = np.array(all_lh)
    all_b = np.array(all_b)
    best_k = np.argmin(all_b)
    best_result = (all_pi[best_k], all_m[best_k], all_S[best_k])

    return best_k + 1, best_result, all_lh, all_b
