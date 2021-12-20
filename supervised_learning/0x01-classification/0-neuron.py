#!/usr/bin/env python3
"""0. Neuron"""


import numpy as np


class Neuron:
    """class Neuron that defines a single neuron performing
    binary classification"""
    def __init__(self, nx):
        """Class constructor.

        nx: is the number of input features to the neuron
            if nx is not an integer, raise a TypeError
            if nx is less than 1, raise a ValueError"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
