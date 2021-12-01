#!/usr/bin/env python3
"""add_matrices2D function"""


def add_matrices2D(mat1, mat2):
    """adds two matrices element-wise"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    newMat = []
    [newMat.append([]) for i in range(len(mat1))]
    for iM in range(len(mat1)):
        for iA in range(len(mat1[0])):
            newMat[iM].append(mat2[iM][iA] + mat1[iM][iA])
    return newMat