#!/usr/bin/env python3
"""Transformer encoder block"""


from asyncio import DatagramProtocol
from tkinter import HIDDEN
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """class EncoderBlock that create an encoder block for transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor

        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        drop_rate - the dropout rate"""
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, 'relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Function that performs embedding

        x - a tensor of shape (batch, input_seq_len, dm)containing
        the input to the encoder block
        training - a boolean to determine if the model is training
        mask - the mask to be applied for multi head attention

        Returns: a tensor of shape (batch, input_seq_len, dm)
        containing the block’s output"""
        attention, weight = self.mha(x, x, x, mask)
        dropout1 = self.dropout1(attention, training=training)
        layernorm = self.layernorm1(x + dropout1)

        hidden = self.dense_hidden(layernorm)
        output = self.dense_output(hidden)

        dropout2 = self.dropout2(output, training=training)
        layernorm2 = self.layernorm2(layernorm + dropout2)

        return layernorm2
