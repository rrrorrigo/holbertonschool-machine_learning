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

    def forward_prop(self, X):
        """Function that calculates the forward propagation of the
        neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data

        Return: the private attributes __A1 and __A2, respectively"""
        x = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.e**(-x))
        x = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.e**(-x))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Function that calculates the cost of the model using logisitic
        regression

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example

        Return: the cost"""
        m = Y.shape[1]
        a = 1.0000001 - A
        x = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(a))
        return x
