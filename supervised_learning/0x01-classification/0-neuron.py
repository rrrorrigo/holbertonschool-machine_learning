#!/usr/bin/env python3
"""Neuron"""


import numpy as np


class Neuron:
    """Neuron class that defines a single neuron performing
    binary classification"""

    def __init__(self, nx):
        """Class constructor

        nx: is the number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
