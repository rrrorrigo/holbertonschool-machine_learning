#!/usr/bin/env python3
"""One Hot Encode"""


import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    return one_hot.argmax(0)