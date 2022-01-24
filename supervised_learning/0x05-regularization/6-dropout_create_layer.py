#!/usr/bin/env python3
"""2-l2 regularization cost"""


import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of a neural network using dropout

    prev: is a tensor containing the output of the previous layer
    n: is the number of nodes the new layer should contain
    activation: is the activation function that should be used on the layer
    keep_prob: is the probability that a node will be kept

    Returns: the output of the new layer"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    l2 = tf.layers.Dropout(1 - keep_prob)
    layer = tf.layers.Dense(
        n, activation, kernel_initializer=init, kernel_regularizer=l2
    )
    return layer(prev)
