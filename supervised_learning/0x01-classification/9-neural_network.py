#!/usr/bin/env python3
"""NeuralNetwork"""


import numpy as np


class NeuralNetwork:
    """NeuralNetwork class that defines a neural network with one hidden layer
    performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor

        nx: is the number of input features to the NeuralNetwork
        nodes: is the number of nodes found in the hidden layer"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b1, self.__b2 = np.zeros(shape=(nodes, 1)), 0
        self.__A1, self.__A2 = 0, 0

    @property
    def W1(self):
        """getter function of attribute W"""
        return self.__W1

    @property
    def b1(self):
        """getter function of attribute b"""
        return self.__b1

    @property
    def A1(self):
        """getter function of attribute A"""
        return self.__A1

    @property
    def W2(self):
        """getter function of attribute W"""
        return self.__W2

    @property
    def b2(self):
        """getter function of attribute B"""
        return self.__b2

    @property
    def A2(self):
        """getter function of attribute A"""
        return self.__A2
