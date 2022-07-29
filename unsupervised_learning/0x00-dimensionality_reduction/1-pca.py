#!/usr/bin/env python3
"""Principal component analysis"""


import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
    version of X"""
    X_meaned = X - np.mean(X, axis=0)

    lsv, sigma, rsv = np.linalg.svd(X_meaned)

    pca_matrix = np.matmul(lsv[..., :ndim], np.diag(sigma[..., :ndim]))

    return pca_matrix