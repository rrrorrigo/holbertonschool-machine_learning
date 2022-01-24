#!/usr/bin/env python3
"""3-one_hot"""


import numpy as np


def one_hot(labels, classes=None):
    """Function that converts a label vector into a one-hot matrix

    The last dimension of the one-hot matrix must be the number of classes

    Returns: the one-hot matrix"""
    l = len(labels)
    b = np.zeros((l, l))
    for i in range(l):
        b[i][labels[i]] = 1
    return b
