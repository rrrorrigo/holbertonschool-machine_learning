#!/usr/bin/env python3
"""2-l2 regularization cost"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Functionthat updates the weights of a neural network with Dropout
    regularization using gradient descent

    Y: is a one-hot numpy.ndarray of shape (classes, m) that contains the
    correct labels for the data
    weights: is a dictionary of the weights and biases of the neural network
    cache: is a dictionary of the outputs and dropout masks of each layer of
    the neural network
    alpha: is the learning rate
    keep_prob: is the probability that a node will be kept
    L: is the number of layers of the network"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        a = cache['A' + str(i - 1)]
        if i == L:
            dz = A - Y
        else:
            x = 1 - (A * A)
            dropout = cache['D' + str(i)]
            dz = np.matmul(W.T, dz) * x
            dz = (dz * dropout) / keep_prob
        W = weights['W' + str(i)]
        dw = np.matmul(dz, a.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
