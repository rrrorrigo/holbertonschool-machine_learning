#!/usr/bin/env python3
"""0. Normalization Constants"""


import numpy as np


def normalize(X, m, s):
    """Function that normalize (standarize) a matrix"""
    return (X - m) / s
