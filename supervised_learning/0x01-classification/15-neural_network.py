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
        self.__W1 = np.random.randn(nodes, nx)
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
        self.__A1 = 1 / (1 + (np.e**-fp1))
        fp2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + (np.e**-fp2))
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
        z2 = A2 - Y
        w2 = np.matmul(z2, A1.T) / m
        b2 = np.sum(z2, axis=1, keepdims=True) / m
        z1 = np.matmul(self.__W2.T, z2) * (A1 * (1 - A1))
        w1 = np.matmul(z1, X.T) / m
        b1 = np.sum(z1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * w2
        self.__b2 -= alpha * b2
        self.__W1 -= alpha * w1
        self.__b1 -= alpha * b1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the neuron

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        iterations: iterations is the number of iterations to train over
        alpha: is the learning rate
        verbose: is a boolean that defines whether or not to print information
            about the training. If True, print Cost after {iteration}
            iterations: {cost} every step iterations
        graph: is a boolean that defines whether or not to graph information
            about the training once the training has completed. If True:
                Plot the training data every step iterations as a blue line
                Label the x-axis as iteration
                Label the y-axis as cost
                Title the plot Training Cost

        Return: the evaluation of the training data after iterations of
            training have occurred"""
        import matplotlib.pyplot as plt

        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        costL = []
        for i in range(iterations + 1):
            cost = self.cost(Y, self.__A2)
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and i % step == 0:
                print('Cost after {} iterations: {}'.format(i, cost))
                if i < iterations:
                    costL.append(cost)
        if graph:
            x = np.arange(0, iterations, step)
            y = costL
            plt.plot(x, y)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
