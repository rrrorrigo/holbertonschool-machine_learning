#!/usr/bin/env python3
"""Neuron"""


import numpy as np


class Neuron:
    """Neuron class that defines a single neuron performing
    binary classification"""

    def __init__(self, nx):
        """Class constructor

        nx: is the number of input features to the neuron"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter function of attribute W"""
        return self.__W

    @property
    def b(self):
        """getter function of attribute W"""
        return self.__b

    @property
    def A(self):
        """getter function of attribute W"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        X: is a numpy.ndarray with shape (nx, m) that contains the input data

        Return: the private attribute A"""
        x = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + np.e**(-x))
        return self.A

    def cost(self, Y, A):
        """Function that calculates the cost of the model using logistic
        regression

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example

        Return: the cost"""
        # To avoid division error I use 1.0000001 - A
        m = Y.shape[1]
        a = 1.0000001 - A
        x = - 1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(a))
        return x

    def evaluate(self, X, Y):
        """Function that evaluates the neuron predictions

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data

        Return: The neuron prediction and the cost of the network respectively
        """
        A = self.forward_prop(X)
        evaluation = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, self.A)
        return evaluation, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Function that calculates the gradient descent on the neuron

        X:  is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        A: is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        alpha: is the learning rate"""
        m = 1 / X.shape[1]
        dz = A - Y
        dw = np.matmul(dz, X.T) * m
        db = np.sum(dz) * m
        self.__W = self.W - (alpha * dw)
        self.__b = self.b - (alpha * db).T

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that trains the neuron

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations: is the number of iterations to train over
        alpha: is the learning rate
        verbose: is a boolean that defines whether or not to print information
        about the training. If True, print Cost after {iteration} iterations:
        {cost} every step iterations:"""
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
            self.gradient_descent(X, Y, self.A, alpha)
            if verbose and (_ % step == 0 or _ == iterations):
                costList.append(self.cost(Y, self.A))
                print('Cost after {} iterations: {}'.format(_, self.cost(Y, self.A)))
            

        if graph:
            import matplotlib.pyplot as plt
            plt.plot(stepArray, costList)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
