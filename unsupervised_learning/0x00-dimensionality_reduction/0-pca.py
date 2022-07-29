#!/usr/bin/env python3
"""Principal component analysis"""


import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    var is the fraction of the variance that the PCA transformation should
    maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s
    original variance"""
    # Apply SVD and extract it into left and right singular vector and sigma
    lsv, sigma, rsv = np.linalg.svd(X)

    diag_sum = np.cumsum(sigma)

    # normalize diagonal sum
    diag_sum =  diag_sum / diag_sum[-1]

    i = np.min(np.where(diag_sum >= var))

    V = rsv.T
    PCA = V[..., :i + 1]

    return PCA
