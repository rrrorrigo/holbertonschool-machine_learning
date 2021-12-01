#!/usr/bin/env python3
"""matrix_transpose function"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    newMatrix = []
    # append empty list to iterate on it afterwards
    [newMatrix.append([]) for i in range(len(matrix[0]))]
    for iArray in range(len(matrix[0])):
        for iMatrix in range(len(matrix)):
            newMatrix[iArray].append(matrix[iMatrix][iArray])
    return newMatrix