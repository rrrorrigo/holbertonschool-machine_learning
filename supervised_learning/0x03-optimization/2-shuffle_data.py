#!/usr/bin/env python3
"""2. Shuffle Data"""


import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two matrices the same way"""
    return np.random.permutation(X), np.random.permutation(Y)
