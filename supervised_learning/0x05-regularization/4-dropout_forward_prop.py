#!/usr/bin/env python3
"""2-l2 regularization cost"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout

    X: is a numpy.ndarray of shape (nx, m) containing the input data
    for the network
    weights: is a dictionary of the weights and biases of the neural network
    L: the number of layers in the network
    keep_prob: is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function

    Returns: a dictionary containing the outputs of each layer and the dropout
    mask used on each layer (see example for format)"""
    output = {}
    output['A0'] = X
    for i in range(L):
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        x = output['A' + str(i)]
        z = np.matmul(W, x) + b
        if i == (L - 1):
            act = (np.exp(z)/np.sum(np.exp(z), axis=0))
        else:
            act = np.tanh(z)
            dropout = np.random.binomial(n=1, p=keep_prob, size=act.shape)
            output['D' + str(i + 1)] = dropout
            act = (act * dropout) / keep_prob
        output['A' + str(i + 1)] = act
    return output
