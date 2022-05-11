#!/usr/bin/env python3
"""Module to work with milticlass classification"""


import numpy as np


def one_hot_decode(one_hot):
    """Function that converts a one-hot matrix into a vector of labels

    one_hot: is a one-hot encoded numpy.ndarray with shape (classes, m)

    Return: a numpy.ndarray with shape (m, ) containing the numeric labels
    for each example, or None on failure"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    return np.array(np.where(one_hot.T)[1])
