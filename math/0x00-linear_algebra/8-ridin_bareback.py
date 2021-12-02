#!/usr/bin/env python3
"""Ridin’ Bareback"""


def mat_mul(mat1, mat2):
    """fuction that performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0 for x in mat2[0]] for y in mat1]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]
    return result
