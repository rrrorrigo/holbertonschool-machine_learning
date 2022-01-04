#!/usr/bin/env python3
"""One Hot Encode"""


import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix

    Y: is a numpy.ndarray with shape (m,) containing numeric class labels
        m is the number of examples
    classes: is the maximum number of classes found in Y"""
    if Y is None or type(Y) is not np.ndarray or type(classes) is not int:
        return None
    try:
        b = np.zeros((classes, Y.shape[0]))
        b[Y, np.arange(Y.shape[0])] = 1
        return b
    except:
        return None