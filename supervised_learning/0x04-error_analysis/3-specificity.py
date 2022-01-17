#!/usr/bin/env python3
"""0. Create Confusion"""


import numpy as np


def specificity(confusion):
    """function that calculates the specificity for each class in a
    confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing the specificity
    of each class"""
    tn = []
    for i in range(confusion.shape[1]):
        temp = np.delete(confusion, i, 0)
        temp = np.delete(temp, i, 1)
        tn.append(sum(sum(temp)))
    fp = np.sum(confusion, axis=0) - np.diagonal(confusion)
    spec = tn / (tn + fp)
    return spec
