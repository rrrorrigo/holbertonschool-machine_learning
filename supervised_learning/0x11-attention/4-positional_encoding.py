#!/usr/bin/env python3
"""Self attention"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """Function that calculates the positional
    encoding for a transformer

    max_seq_len is an integer representing the maximum
    sequence length
    dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm)
    containing the positional encoding vectors"""
    x = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in range(int(dm / 2)):
            denominator = np.power(10000, 2*i/dm)
            x[k, 2*i] = np.sin(k/denominator)
            x[k, 2*i+1] = np.cos(k/denominator)
    return x
