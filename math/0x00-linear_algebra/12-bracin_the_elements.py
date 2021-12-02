#!/usr/bin/env python3
"""Bracing The Elements"""


def np_elementwise(mat1, mat2):
    """function that performs element-wise addition, subtraction,
    multiplication, and division"""
    import numpy as np
    return (np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2))
