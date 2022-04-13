#!/usr/bin/env python3
"""Flip me over"""


def matrix_transpose(matrix):
    """Function that returns the transpose of a 2D matrix"""
    lenMatrix = len(matrix)
    lenArray = len(matrix[0])
    rmatrix = []
    for i in range(lenArray):
        array = []
        for n in range(lenMatrix):
            array.append(matrix[n][i])
        rmatrix.append(array)
    return rmatrix
