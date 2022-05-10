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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for n in range(len(layers)):
            if layers[n] < 1 or type(layers[n]) is not int:
                raise TypeError('layers must be a list of positive integers')
            w = 'W' + str(n + 1)
            b = 'b' + str(n + 1)
            if n == 0:
                layerPrev = nx
            else:
                layerPrev = layers[n - 1]
            sqrt = np.sqrt(2 / layerPrev)
            self.__weights[w] = np.random.randn(layers[n], layerPrev) * sqrt
            self.__weights[b] = np.zeros(shape=(layers[n], 1))

    @property
    def L(self):
        """Getter function of private attribute n"""
        return self.__L

    @property
    def weights(self):
        """Getter function of private attribute weights"""
        return self.__weights

    @property
    def cache(self):
        """Getter function of private attribute cache"""
        return self.__cache

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the
        neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data

        Return: output of the neural network and the cache, respectively"""
        self.__cache['A0'] = X
        for n in range(self.L):
            a = 'A' + str(n)
            aNext = 'A' + str(n + 1)
            w = 'W' + str(n + 1)
            b = 'b' + str(n + 1)
            x = np.matmul(self.weights[w], self.cache[a]) + self.weights[b]
            self.__cache[aNext] = 1 / (1 + np.e**(-x))
        return self.__cache[aNext], self.cache

    def cost(self, Y, A):
        """Function that calculates the cost of the model using logistic
        regression

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example

        Return: the cost"""
        m = Y.shape[1]
        a = 1.0000001 - A
        x = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(a))
        return x
