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
        if type(layers) is not list or len(layers) == 0 or np.any(arr < 0):
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if i == 0:
                self.__weights['W1'] = np.random.randn(
                    layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(
                        2 / (layers[i - 1] + layers[i]))
            self.__weights['b' + str(i + 1)] = np.zeros(
                layers[i]).reshape(layers[i], 1)

    @property
    def L(self):
        """getter method for attirbute L"""
        return self.__L

    @property
    def cache(self):
        """getter method for attirbute cache"""
        return self.__cache

    @property
    def weights(self):
        """getter method for attirbute weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        """
        self.__cache['A0'] = X
        for i in range(self.L):
            n = str(i + 1)
            self.__cache['A' + n] = 1 / (
                1 + (np.e**-(np.matmul(
                    self.weights['W' + n], self.cache['A' + str(i)]
                    ) + self.weights['b' + n])))
        return self.__cache['A' + n], self.__cache

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
        """Evaluates the neural network’s predictions

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data

        Return: the neuron’s prediction and the cost of the network,
        respectively"""
        self.forward_prop(X)
        predict = np.where(self.cache['A' + str(self.L)] >= 0.5, 1, 0)
        cost = self.cost(Y, self.cache['A' + str(self.L)])
        return (predict, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        cache: is a dictionary containing all the intermediary values of
            the network
        alpha: is the learning rate"""
        m = Y.shape[1]
        for i in range(self.L - 1, 0, -1):
            A = cache['A' + str(i + 1)]
            a = cache['A' + str(i)]
            x = A * (1 - A)
            if i == self.L - 1:
                dz = A - Y
                W = self.__weights['W' + str(i + 1)]
            else:
                dz = np.matmul(W.T, dz) * x
            dw = np.matmul(a, dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W' + str(i + 1)] = self.__weights['W' + str(i + 1)] - alpha * dw.T
            self.__weights['b' + str(i + 1)] = self.__weights['b' + str(i + 1)] - alpha * db
