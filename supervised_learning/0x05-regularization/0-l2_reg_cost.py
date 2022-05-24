#!/usr/bin/env python3
""""""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network with L2
    regularization:

    cost: is the cost of the network without L2 regularization
    lambtha: is the regularization parameter
    weights: is a dictionary of the weights and biases (numpy.ndarrays)
    of the neural network
    L: is the number of layers in the neural network
    m: is the number of data points used

    Returns: the cost of the network accounting for L2 regularization"""
    for i in range(L):
        frobenius_norm = np.linalg.norm(weights['W' + str(i + 1)])
        cost += lambtha / 2 / m * frobenius_norm
    return cost
