#!/usr/bin/env python3
"""NeuralNetwork"""


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
        self.__W1 = np.random.randn(1, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # output of each neuron
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter funciton of attribute W1"""
        return self.__W1

    @property
    def W2(self):
        """getter funciton of attribute W2"""
        return self.__W2
        
    @property
    def b1(self):
        """getter funciton of attribute b1"""
        return self.__b1

    @property
    def b2(self):
        """getter funciton of attribute b2"""
        return self.__b2

    @property
    def A1(self):
        """getter funciton of attribute A1"""
        return self.__A1

    @property
    def A2(self):
        """getter funciton of attribute A2"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        """
        fp1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid = lambda x: 1 / (1 + (np.e**-x))
        self.__A1 = sigmoid(fp1)
        fp2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(fp2)
        return self.__A1, self.__A2

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
        predict = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return (predict, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        A1: is the output of the hidden layer
        A2: is the predicted output
        alpha: is the learning rate"""
        m = X.shape[1]  # number of trainig examples
        z1 = A1 - Y
        z2 = A2 - Y
        w1 = np.matmul(X, z1.T) / m
        b1 = np.sum(z1) / m
        w2 = np.matmul(X, z2.T) / m
        b2 = np.sum(z2) / m
        # gradient descent formula for Weight and bias
        updtAttributes = lambda x, y: x - (alpha * y).T
        self.__W1 = updtAttributes(self.__W1, w1)
        self.__b1 = updtAttributes(self.__b1, b1)
        self.__W2 = updtAttributes(self.__W2, w2)
        self.__b2 = updtAttributes(self.__b2, b2)
