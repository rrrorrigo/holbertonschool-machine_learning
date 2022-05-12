#!/usr/bin/env python3
"""Create placeholder"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function that create a layer

    prev: is the tensor output of the previous layer
    n: is the number of nodes in the layer to create
    activation: is the activation function that the layer should be

    Return: the tensor output of the layer"""
    heetal = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    return tf.keras.layers.Dense(n, activation, kernel_initializer=heetal)(prev)
