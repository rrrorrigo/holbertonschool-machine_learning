#!/usr/bin/env python3
"""Gated recurrent unit cell"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN

    rnn_cells is a list of RNNCell instances of length l that will
    be used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a
    numpy.ndarray of shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: h_next, Y
        h_next is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs"""
    h_next = np.array(([h_0]))
    h_next = np.repeat(h_next, X.shape[0] + 1, axis=0)

    for i in range(X.shape[0]):
        for a_layer, cell in enumerate(rnn_cells):
            parameter = X[i] if a_layer == 0 else h_prev
            h_prev, y = cell.forward(h_next[i, a_layer], parameter)

            h_next[i + 1, a_layer] = h_prev

            if (i != 0):
                Y[i] = y
            else:
                Y = np.array([y])
                Y = np.repeat(Y, X.shape[0], axis=0)
    return h_next, y
