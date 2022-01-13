#!/usr/bin/env python3
"""7. RMSProp"""


import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for a neural network
    in tensorflow

    prev: is the activated output of the previous layer
    n: is the number of nodes in the layer to be created
    activation: is the activation function that should be used on the output
        of the layer
    you should use the tf.keras.layers.Dense layer as the base layer with
        kernal initializer
        tf.keras.initializers.VarianceScaling(mode='fan_avg')
    your layer should incorporate two trainable parameters, gamma and beta,
        initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-8

    Returns: a tensor of the activated output for the layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layers = tf.layers.Dense(units=n, kernel_initializer=init)
    epsilon = 1e-8
    previousOutput = layers(prev)

    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")
    mean, variance = tf.nn.moments(previousOutput, axes=0)
    batch = tf.nn.batch_normalization(previousOutput, mean, variance,
                                      offset=beta, scale=gamma,
                                      variance_epsilon=epsilon)
    if not activation:
        return batch
    return activation(batch)
