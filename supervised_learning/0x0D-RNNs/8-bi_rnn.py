#!/usr/bin/env python3
"""Bidirectional cell of Recurrent neural network"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Function that performs forward propagation for a bidirectional RNN

    bi_cell is an instance of BidirectinalCell that will be used for the
    forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction, given as
    a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction, given as
    a numpy.ndarray of shape (m, h)

    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs"""
    t, m, i = X.shape
    h = h_0.shape[1]
    H_for = np.zeros((t, m, h))
    H_back = np.zeros((t, m, h))
    h_forward = h_0
    h_backward = h_t

    for i in range(t):
        x_forward = X[i]
        x_backward = X[-(i + 1)]

        h_forward = bi_cell.forward(h_forward, x_forward)
        h_backward = bi_cell.backward(h_backward, x_backward)

        H_for[i] = h_forward
        H_back[-(i + 1)] = h_backward

    H = np.concatenate((H_for, H_back), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
