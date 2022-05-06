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
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter function of attribute W"""
        return self.__W

    @property
    def b(self):
        """getter function of attribute W"""
        return self.__b

    @property
    def A(self):
        """getter function of attribute W"""
        return self.__A