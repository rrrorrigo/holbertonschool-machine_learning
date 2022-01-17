#!/usr/bin/env python3
"""0. Create Confusion"""


import numpy as np


def sensitivity(confusion):
    """Funtion that calculates the sensitivity for each class in a confusion
    matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of
    each class"""
    tpfn = np.sum(confusion, axis=1)
    tp = np.diagonal(confusion)
    sens = tp / tpfn
    return sens
