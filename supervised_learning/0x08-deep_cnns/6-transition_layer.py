#!/usr/bin/env python3
"""
    Transition Layer - Densely Connected Convolutional Networks
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    Your code should implement compression as used in DenseNet-C
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of filters
    within the output, respectively
    """
    init = K.initializers.HeNormal()
    outputPrev = X
    batchNorm = K.layers.BatchNormalization()(outputPrev)
    activation = K.layers.Activation('relu')(batchNorm)
    nFilt = int(nb_filters * compression)
    conv = K.layers.Conv2D(nFilt, 1,
                           padding='same',
                           kernel_initializer=init)(activation)
    avg_pooling = K.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(conv)

    return avg_pooling, nFilt
