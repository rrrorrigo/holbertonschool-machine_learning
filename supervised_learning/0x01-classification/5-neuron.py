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
        self.__W = np.random.randn(1, nx)  # weight of vector
        self.__b = 0  # bias of each neuron
        self.__A = 0  # Activated output of each neuron

    @property
    def W(self):
        """getter function of attribute W"""
        return self.__W

    @property
    def b(self):
        """getter function of attribute b"""
        return self.__b

    @property
    def A(self):
        """getter function of attribute A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        forwardPropagation = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + (np.e**-forwardPropagation))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example"""
        m = Y.shape[1]
        a = 1.0000001 - A
        y = 1 - Y
        totalCost = -(1 / m) * np.sum(Y * np.log(A) + y * np.log(a))
        return totalCost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
            nx is the number of input features to the neuron
            m is the number of examples
        Y:  is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data

        Return:  the neuron’s prediction and the cost of the network,
        respectively."""
        self.forward_prop(X)
        predict = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return (predict, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        alpha: is the learning rate"""
        m = X.shape[1]  # number of trainig examples
        z = A - Y
        w = np.matmul(X, z.T) / m
        b = np.sum(z) / m
        self.__W = self.__W - alpha * w
        self.__b = self.__b - alpha * b
