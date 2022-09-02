#!/usr/bin/env python3
"""Gated recurrent unit cell"""


import numpy as np


class LSTMCell:
    """Class that represents an long short-term memory"""

    def __init__(self, i, h, o):
        """Class constructor

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo, Wy,
        bf, bu, bc, bo, by that represent the weights and biases of the cell
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs"""
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Function that performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the
        data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the
        previous cell state

        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell"""
        Whh = np.concatenate((h_prev, x_t), axis=1)

        forget_gate = np.dot(Whh, self.Wf) + self.bf
        forget_gate = 1 / (1 + np.exp(-forget_gate))

        update_gate = np.dot(Whh, self.Wu) + self.bu
        update_gate = 1 / (1 + np.exp(-update_gate))

        output_gate = np.dot(Whh, self.Wo) + self.bo
        output_gate = 1 / (1 + np.exp(-output_gate))

        intermediate_state = np.tanh(np.dot(Whh, self.Wc) + self.bc)
        c_next = update_gate * intermediate_state + forget_gate * c_prev
        h_next = output_gate * np.tanh(c_next)

        h_y = np.dot(h_next, self.Wy) + self.by

        y = np.exp(h_y) / np.sum(np.exp(h_y), axis=1, keepdims=True)

        return h_next, c_next, y
