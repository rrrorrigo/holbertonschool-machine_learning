#!/usr/bin/env python3
"""Recurrent neural network encoder"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class that inherits from tensorflow.keras.layers.Layer
    to encode for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """class constructor

        vocab is an integer representing the size of the input vocabulary
        embedding is an integer representing the dimensionality of
        the embedding vector
        units is an integer representing the number of hidden units
        in the RNN cell
        batch is an integer representing the batch size"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Function that initialize hidden state"""
        initializer = tf.keras.initializers.zeros()
        values = initializer(shape=(self.batch, self.units))
        return values

    def call(self, x, initial):
        """Function that performs embedding

        x: is a tensor of shape (batch, input_seq_len) containing the input to
        the encoder layer as word indices within the vocabulary
        initial: is a tensor of shape (batch, units) containing
        the initial hidden state

        Returns: outputs, hidden
            outputs is a tensor of shape (batch, input_seq_len, units)
            containing the outputs of the encoder
            hidden is a tensor of shape (batch, units) containing the
            last hidden state of the encoder"""
        input = self.embedding(x)
        output, hidden = self.gru(input, initial_state=initial)
        return output, hidden
