#!/usr/bin/env python3
"""cat_matrices2D function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis"""
    mat1Copy = [row[:] for row in mat1]
    mat2Copy = [row[:] for row in mat2]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat2)):
            mat1Copy.append(mat2Copy[i])
        return mat1Copy
    if axis == 1 and len(mat1) == len(mat2):
        newMat = []
        for i in range(len(mat1)):
            newMat.append(mat1Copy[i] + mat2Copy[i])
        return newMat
    else:
        return None
