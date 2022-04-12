#!/usr/bin/env python3
"""
    Dense Block - Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense
    Block and the number of filters within the concatenated outputs,
    respectively
    """
    init = K.initializers.HeNormal()
    output_prev = X
    for _ in range(layers):
        batch_norm = K.layers.BatchNormalization()(output_prev)
        activation = K.layers.Activation('relu')(batch_norm)
        conv2 = K.layers.Conv2D(growth_rate * 4,
                                1,
                                padding='same',
                                kernel_initializer=init)(activation)
        batch_norm_1 = K.layers.BatchNormalization()(conv2)
        activation_1 = K.layers.Activation('relu')(batch_norm_1)
        conv2_1 = K.layers.Conv2D(growth_rate, 3,
                                  padding='same',
                                  kernel_initializer=init)(activation_1)
        concat_output_prev = K.layers.Concatenate()([output_prev, conv2_1])
        output_prev = concat_output_prev
        nb_filters += growth_rate

    return concat_output_prev, nb_filters
