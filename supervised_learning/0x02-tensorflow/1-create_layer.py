#!/usr/bin/env python3
"""0. Placeholders"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function that create a layer

    prev: is the tensor output of the previous layer
    n: is the number of nodes in the layer to create
    activation: is the activation function that the layer should use

    Returns: the tensor output of the layer"""
    he_et_al = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=he_et_al, name="layer")
    return layer(prev)
