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

    def evaluate(self, X, Y):
        """Function that evaluates the neural network predictions

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data

        Return: the neuron prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)[1]
        evaluation = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return evaluation, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Function that aclculates one pass of gradient descent on the
        neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A1: is the output of the hidden layer
        A2: is the predicted output
        alpha: is the learning rate"""
        m = 1 / X.shape[1]
        dz2 = A2 - Y
        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dw = np.matmul(dz1, X.T) * m
        self.__W1 = self.W1 - (alpha * dw)
        self.__b1 = self.b1 - (alpha * dz1.mean(axis=1, keepdims=True))
        dw = np.matmul(dz2, A1.T) * m
        self.__W2 = self.W2 - (alpha * dw)
        self.__b2 = self.b2 - (alpha * dz2.mean(axis=1, keepdims=True))

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that trains the neural network

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations: is the number of iterations to train over
        alpha: is the learning rate
        verbose: is a boolean that defines whether or not to print information
        about the training
        graph: is a boolean that defines whether or not to graph information
        about the training once the training has completed

        Return: the evaluation of the training data after iterations of
        training have occurred"""
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        costList = []
        stepArray = np.arange(0, iterations + 1, step)
        for _ in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)
            if verbose and (_ % step == 0 or _ == iterations):
                costList.append(self.cost(Y, self.A2))
                print('Cost after {} iterations: {}'
                      .format(_, self.cost(Y, self.A2)))
        if graph:
            import matplotlib.pyplot as plt
            plt.plot(stepArray, costList)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
