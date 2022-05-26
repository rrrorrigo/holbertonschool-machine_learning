#!/usr/bin/env python3
"""Dropout forward propagation"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout:

    X: is a numpy.ndarray of shape (nx, m) containing the input data for the
    network
        nx: is the number of input features
        m: is the number of data points
    weights: is a dictionary of the weights and biases of the neural network
    L: the number of layers in the network
    keep_prob: is the probability that a node will be kept

    Returns: a dictionary containing the outputs of each layer and the dropout
    mask used on each layer (see example for format)"""
    cache = {}
    cache['A0'] = X
    for n in range(L):
        a = 'A' + str(n)
        aNext = 'A' + str(n + 1)
        w = 'W' + str(n + 1)
        b = 'b' + str(n + 1)

        x = np.matmul(weights[w], cache[a]) + weights[b]
        dropout = np.where(np.random.rand(x.shape[0],
                                          x.shape[1]) < keep_prob, 1, 0)
        if n < L - 1:
            before_dropout = np.tanh(x)
            dropout_result = np.multiply(before_dropout, dropout) / keep_prob
            cache['D' + str(n + 1)] = dropout
        else:
            before_dropout = np.exp(x) / np.sum(np.exp(x), axis=0)
            dropout_result = before_dropout
        cache[aNext] = dropout_result
    return cache
