#!/usr/bin/env python3
def matrix_shape(matrix):
        """Function that calculates the shape of a matrix:"""
        shape = []
        while type(matrix) == list:
                for arr in matrix:
                        shape.append(len(matrix))
                        matrix = arr
                        break
        return shape