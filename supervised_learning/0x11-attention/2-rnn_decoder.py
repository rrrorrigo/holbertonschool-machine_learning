#!/usr/bin/env python3
"""Recurrent Neural Network decoder"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Class that decode machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """Class constructor

        vocab is an integer representing the size of the
        output vocabulary
        embedding is an integer representing the dimensionality
        of the embedding vector
        units is an integer representing the number of
        hidden units in the RNN cell
        batch is an integer representing the batch size
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       kernel_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.Attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """Function that performs embedding

        x is a tensor of shape (batch, 1) containing the previous
        word in the target sequence as an index of the target vocabulary
        s_prev is a tensor of shape (batch, units) containing the
        previous decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder"""
        context, weights = self.Attention(s_prev, hidden_states)
        layer = self.embedding(x)
        layer = tf.concat([tf.expand_dims(context, axis=1), layer], axis=-1)
        sequence, state = self.gru(layer)
        sequence = tf.reshape(sequence, (-1, sequence.shape[2]))

        y = self.F(sequence)
        return y, state
