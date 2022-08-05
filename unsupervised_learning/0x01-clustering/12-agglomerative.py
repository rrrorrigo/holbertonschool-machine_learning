#!/usr/bin/env python3
"""scipy"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Function that performs agglomerative clustering on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color

    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
    indices for each data point"""
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    fcluster = hierarchy.fcluster(linkage, dist, criterion='distance')

    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()

    return fcluster
