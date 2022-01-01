#!/usr/bin/env python3
"""DeepNeuralNetwork"""


import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """Class constructor

        nx: is the number of input features
        layers: is a list representing the number of nodes in each layer
            of the network"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        arr = np.array(layers)
        if type(layers) is not list or len(layers) == 0 or np.any(arr < 0)\
            or np.any(arr is int):
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(
                        2 / (layers[i - 1] + layers[i]))
            self.weights['b' + str(i + 1)] = np.zeros(
                layers[i]).reshape(layers[i], 1)
