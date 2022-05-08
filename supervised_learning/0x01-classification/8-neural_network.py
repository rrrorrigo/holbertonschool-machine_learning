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
        self.W1 = np.random.normal(size=(nodes, nx))
        self.W2 = np.random.normal(size=(1, nodes))
        self.b1, self.b2 = 0, 0
        self.A1, self.A2 = 0, 0