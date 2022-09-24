#!/usr/bin/env python3
"""Self attention"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Function that calculates the scaled dot product attention:

    Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
     containing the query matrix
    K is a tensor with its last two dimensions as (..., seq_len_v, dk)
     containing the key matrix
    V is a tensor with its last two dimensions as (..., seq_len_v, dv)
     containing the value matrix
    mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v)
    containing the optional mask, or defaulted to None
        if mask is not None, multiply -1e9 to the mask and add it to
        the scaled matrix multiplication

    Returns: output, weights
        outputa tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention
        weights a tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) containing the attention weights"""
    qk = tf.matmul(Q, K, transpose_b=True)
    
    K_float32 = tf.cast(tf.shape(K)[-1], tf.float32)

    scaled = qk / tf.math.sqrt(K_float32)

    if mask is not None:
        scaled += (mask * -1e9)

    weight = tf.nn.softmax(scaled, axis=-1)

    output = tf.matmul(weight, V)

    return output, weight
