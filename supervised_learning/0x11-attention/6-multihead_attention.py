#!/usr/bin/env python3
"""Multihead Attention"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class that performs multi head attention"""

    def __init__(self, dm, h):
        """class constructor

        dm is an integer representing the dimensionality of the model
        h is an integer representing the number of heads
        dm is divisible by h"""
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """Function that performs embedding

        Q is a tensor of shape (batch, seq_len_q, dk) containing the
        input to generate the query matrix
        K is a tensor of shape (batch, seq_len_v, dk) containing the
        input to generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dv) containing the
        input to generate the value matrix
        mask is always None

        Returns: output, weights
            outputa tensor with its last two dimensions as (..., seq_len_q, dm)
            containing the scaled dot product attention
            weights a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights"""
        batch = tf.shape(Q)[0]

        q = self.split_h(self.Wq(Q), batch)
        k = self.split_h(self.Wk(K), batch)
        v = self.split_h(self.Wv(V), batch)

        y, weight = sdp_attention(q, k, v, mask)

        y = tf.transpose(y, perm=[0, 2, 1, 3])

        y = tf.reshape(y, (batch, -1, self.dm))

        Y = self.linear(y)

        return Y, weight

    def split_h(self, x, batch):
        """Function that splits data"""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        return x
