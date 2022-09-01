#!/usr/bin/env python3
"""Recurrent neural network"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """Function that performs forward propagation for a simple RNN

    rnn_cell is an instance of RNNCell that will be used for
    the forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state

    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs"""
    t, m, i = X.shape
    m, h = h_0.shape
    y_size = rnn_cell.by.shape[1]

    h_next = np.zeros((t + 1, m, h))
    y = np.zeros((t, m, y_size))

    h_next[0, ...], y[0, ...] = rnn_cell.forward(h_0, X[0, :, :])

    for i in range(1, t):
        h_next[i, ...], y[i, ...] = rnn_cell.forward(h_next[i, ...], X[i, ...])
    return h_next, y
