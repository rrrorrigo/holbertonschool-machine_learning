#!/usr/bin/env python3
"""NeuralNetwork"""


from typing import Type
import numpy as np


class NeuralNetwork:
    """defines a neural network with one hidden layer
    performing binary classification"""
    def __init__(self, nx, nodes):
        """class constructor.

        nx:  is the number of input features
            If nx is not an integer, raise a TypeError with the exception:
                nx must be an integer
            If nx is less than 1, raise a ValueError with the exception:
                nx must be a positive integer
        nodes: is the number of nodes found in the hidden layer
            If nodes is not an integer, raise a TypeError with the exception:
                nodes must be an integer
            If nodes is less than 1, raise a ValueError with the exception:
                nodes must be a positive integer"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.W1 = np.random.randn(1, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        # output of each neuron
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
