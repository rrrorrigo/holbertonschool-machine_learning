#!/usr/bin/env python3
"""Probability Distribution Function"""


import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density function of a
    Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the data points whose
    PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of
    the distribution
    numpy.ndarray.diagonal
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    n, d = X.shape
    f = 1 / np.sqrt(((2*np.pi)**d) * np.linalg.det(S))
    Xm = (X - m).T
    g = np.exp(-0.5 * np.sum(np.matmul(np.linalg.inv(S), Xm) * Xm, axis=0))
    P = f * g

    return np.maximum(P, 1e-300)
