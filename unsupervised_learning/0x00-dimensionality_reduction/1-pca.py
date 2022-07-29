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
    X_meaned = X - np.mean(X , axis = 0)
     
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    # arranging them in descending order of their Eigenvalue will automatically
    # arrange the principal component in descending order of their variability
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    eigenvector_subset = sorted_eigenvectors[:,0:ndim]
     
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced