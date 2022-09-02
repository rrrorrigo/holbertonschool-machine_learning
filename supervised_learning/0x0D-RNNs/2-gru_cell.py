#!/usr/bin/env python3
"""Gated recurrent unit cell"""


import numpy as np


class GRUCell:
    """class that represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """Class constructor

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz,
        br, bh, by that represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output"""
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function that performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data
        input for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state

        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell"""
        Whh = np.concatenate((h_prev, x_t), axis=1)

        update_gate = np.dot(Whh, self.Wz) + self.bz
        update_gate = 1 / (1 + np.exp(-update_gate))

        reset_gate = np.dot(Whh, self.Wr) + self.br
        reset_gate = 1 / (1 + np.exp(-reset_gate))

        Whh = np.concatenate((reset_gate * h_prev, x_t), axis=1)
        hidden_state = np.tanh(np.dot(Whh, self.Wh) + self.bh)
        h_next = update_gate * hidden_state + (1 - update_gate) * h_prev

        h_y = np.dot(h_next, self.Wy) + self.by
        y = np.exp(h_y) / np.sum(np.exp(h_y), axis=1, keepdims=True)

        return h_next, y
