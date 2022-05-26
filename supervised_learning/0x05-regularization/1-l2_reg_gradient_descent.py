#!/usr/bin/env python3
"""Gradient descent with L2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ that updates the weights and biases of a neural network using gradient
    descent with L2 regularization:

    Y: is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
        classes: is the number of classes
        m: is the number of data points
    weights: is a dictionary of the weights and biases of the neural network
    cache: is a dictionary of the outputs of each layer of the neural network
    alpha: is the learning rate
    lambtha: is the L2 regularization parameter
    L: is the number of layers of the network"""
    m = Y.shape[1]
    la = L
    a = 'A' + str(la)
    W = 'W' + str(la)
    b = 'b' + str(la)
    dz = cache[a] - Y
    dw = (np.dot(cache['A' + str(la - 1)], dz.T) / m).T
    dw = dw + (lambtha / m) * weights[W]
    db = np.sum(dz, axis=1, keepdims=True) / m
    weights[W] = weights[W] - alpha * dw
    weights[b] = weights[b] - alpha * db

    for la in range(L - 1, 0, -1):
        a = 'A' + str(la)
        W = 'W' + str(la)
        b = 'b' + str(la)
        wNext = 'W' + str(la + 1)
        aNext = 'A' + str(la - 1)
        g = cache[a] * (1 - cache[a])
        dz = np.dot(weights[wNext].T, dz) * g
        dw = (np.dot(cache[aNext], dz.T) / m).T + ((lambtha / m) * weights[W])
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights[W] = weights[W] - alpha * dw
        weights[b] = weights[b] - alpha * db
