#!/usr/bin/env python3
"""Bidirectional cell of Recurrent neural network"""


import numpy as np


class BidirectionalCell:
    """Class that represent a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Class constructor

        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy,
        bhf, bhb, by that represent the weights and biases of the cell
            Whf and bhfare for the hidden states in the forward direction
            Whb and bhbare for the hidden states in the backward direction
            Wy and byare for the outputs"""
        self.Whf = np.random.normal(size=(h + i, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function that calculates the hidden state in the forward direction
        for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state

        Returns: h_next, the next hidden state"""
        Whh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(Whh, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Function that calculates the hidden state in the backward
        direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data
        input for the cell
            m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing the
        next hidden state

        Returns: h_pev, the previous hidden state"""
        Whh = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(Whh, self.Whb) + self.bhb)

        return h_prev
