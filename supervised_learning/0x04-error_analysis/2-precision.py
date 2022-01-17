#!/usr/bin/env python3
"""0. Create Confusion"""


import numpy as np


def precision(confusion):
    """Function that calculates the precision for each class in a confusion
    matrix

    confusion: is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels

    Returns: a numpy.ndarray of shape (classes,) containing the precision of
    each class"""
    tp = np.diagonal(confusion)
    tpfp = np.sum(confusion, axis=0)
    return tp / tpfp
