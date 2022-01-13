#!/usr/bin/env python3
"""0. Normalization Constants"""


import numpy as np


def normalization_constants(X):
    """function that calculates the normalization
    (standardization) constants of a matrix"""
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    std_dev = np.sqrt(np.sum((X - mean) ** 2, axis=0) / m)
    return mean, std_dev