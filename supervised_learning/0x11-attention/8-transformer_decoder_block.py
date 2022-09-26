#!/usr/bin/env python3
"""Transformer decoder block"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block class that create an encoder block for a transoformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor

        dm - the dimensionality of the model
        h - the number of heads
        hidden - the number of hidden units in the fully connected layer
        drop_rate - the dropout rate"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, 'relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Function that performs embedding

        x - a tensor of shape (batch, target_seq_len, dm)containing the input
        to the decoder block
        encoder_output - a tensor of shape (batch, input_seq_len, dm)containing
        the output of the encoder
        training - a boolean to determine if the model is training
        look_ahead_mask - the mask to be applied to the first multi head
        attention layer
        padding_mask - the mask to be applied to the second multi head
        attention layer

        Returns: a tensor of shape (batch, target_seq_len, dm) containing
        the block’s output"""
        attention, weight = self.mha1(x, x, x, look_ahead_mask)
        dropout = self.dropout1(attention, training=training)
        layernorm = self.layernorm1(dropout + x)

        attention2, weight = self.mha2(layernorm, encoder_output,
                                       encoder_output, padding_mask)
        dropout2 = self.dropout2(attention2, training=training)
        layernorm2 = self.layernorm2(dropout2 + layernorm)

        dropout3 = self.dropout3(layernorm2, training=training)
        layernorm3 = self.layernorm3(dropout3 + layernorm2)

        return layernorm3
