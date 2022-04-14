#!/usr/bin/env python3
"""Saddle Up"""


def np_slice(matrix=[], axes={}):
    """function that slices a matrix along specific axes"""
    import numpy as np
    slc = np.asarray(matrix)
    newMatrix = []
    for idx in range(len(slc)):
        for k, v in axes.items():
            lenValue = len(v)
            start = v[0] if lenValue > 0 else None
            stop = v[1] if lenValue > 1 else None
            step = v[2] if lenValue > 2 else None
            axis = k
            newMatrix.append(slc[idx][start:stop:step])
    return np.asarray(newMatrix)
