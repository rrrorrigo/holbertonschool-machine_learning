#!/usr/bin/env python3
"""0. L2 Regularization Cost"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function  that updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Y: is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
    weights: is a dictionary of the weights and biases of the neural network
    cache: is a dictionary of the outputs of each layer of the neural network
    alpha: is the learning rate
    lambtha: is the L2 regularization parameter
    L: is the number of layers of the network"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        a = cache['A' + str(i - 1)]
        if i == L:
            dz = A - Y
        else:
            x = 1 - (A * A)
            dz = np.matmul(W.T, dz) * x
        W = weights['W' + str(i)]
        dw = np.matmul(dz, a.T) / m + (lambtha / m * W)
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
