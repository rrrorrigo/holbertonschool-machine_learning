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

    def evaluate(self, X, Y):
        """Function that evaluates the neural network prediction

        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data

        Return: The neuron prediction and the cost of the network, respectively
        """
        A = self.forward_prop(X)[0]
        evaluation = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return evaluation, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Function that calculates one pass of gradient descent on the neural
        network

        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        cache: is a dictionary containing all the intermediary values of the
        network
        alpha: is the learning rate"""
        m = Y.shape[1]
        weight = self.__weights.copy()
        la = self.__L
        a = 'A' + str(la)
        W = 'W' + str(la)
        b = 'b' + str(la)
        dz = self.__cache[a] - Y
        dw = np.dot(self.__cache['A' + str(la - 1)], dz.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        self.__weights[W] = weight[W] - alpha * dw.T
        self.__weights[b] = weight[b] - alpha * db

        for la in range(self.__L - 1, 0, -1):
            a = 'A' + str(la)
            W = 'W' + str(la)
            b = 'b' + str(la)
            wNext = 'W' + str(la + 1)
            aNext = 'A' + str(la - 1)
            g = self.__cache[a] * (1 - self.__cache[a])
            dz = np.dot(weight[wNext].T, dz) * g
            dw = np.dot(self.__cache[aNext], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            self.__weights[W] = weight[W] - alpha * dw.T
            self.__weights[b] = weight[b] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Function that Trains the deep neural network
        X: is a numpy.ndarray with shape (nx, m) that contains the input data
        Y: is a numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        iterations: is the number of iterations to train over
        alpha: is the learning rate"""
        import matplotlib.pyplot as plt
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step < 1 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        steps = []
        costs = []
        for cont in range(iterations + 1):
            self.forward_prop(X)
            cache = self.__cache
            self.gradient_descent(Y, cache, alpha)
            if cont == iterations or cont % step == 0:
                cost = self.cost(Y, self.__cache['A' + str(self.__L)])
                if verbose:
                    print('Cost after {} iterations: {}'.format(cont, cost))
                if graph:
                    costs.append(cost)
                    steps.append(cont)
        if graph:
            plt.plot(steps, costs)
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.show()
        return self.evaluate(X, Y)
