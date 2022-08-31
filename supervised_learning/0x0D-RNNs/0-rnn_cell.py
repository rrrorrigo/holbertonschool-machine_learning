#!/usr/bin/env python3
"""Recurrent neural network"""


import numpy as np


class RNNCell:
    """class that represent a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """Class constructor

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh, by that
        represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        The weights will be initialized using a random normal
        distribution in the order listed above
        The biases will be initialized as zeros"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data
        input for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state

        The output of the cell use a softmax activation function

        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell"""
        # solution explained https://victorzhou.com/blog/intro-to-rnns/
        Whx = self.Wh
        Why = self.Wy
        bh = self.bh
        by = self.by

        Whh = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.dot(Whh, Whx) + bh)
        h_y = np.dot(h_next, Why) + by
        y = np.exp(h_y) / np.sum(np.exp(h_y), axis=1, keepdims=True)

        return h_next, y
