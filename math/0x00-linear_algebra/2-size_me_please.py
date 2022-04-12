#!/usr/bin/env python3
"""Size me please"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix"""
    rlist = []
    while type(matrix) is list:
        for arr in matrix:
            rlist.append(len(matrix))
            matrix = arr
            break
    return rlist