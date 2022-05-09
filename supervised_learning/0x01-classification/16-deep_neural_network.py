#!/usr/bin/env python3
"""DeepNeuralNetwork"""


import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork class that defines a neural network with one hidden
    layer performing binary classification"""

    def __init__(self, nx, layers):
        """Class constructor

        nx: is the number of input features to the DeepNeuralNetwork
        layers: is the number of layers found in the hidden layer"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(len(layers)):
            if layers[l] < 1 or type(layers[l]) is not int:
                raise TypeError('layers must be a list of positive integers')
            w = 'W' + str(l + 1)
            b = 'b' + str(l + 1)
            if l == 0:
                layerPrev = nx
            else:
                layerPrev = layers[l]
            sqrt = np.sqrt(2 / layerPrev)
            self.weights[w] = np.random.randn(layers[l], layerPrev) * sqrt
            self.weights[b] = np.zeros(shape=(layers[l], 1))
