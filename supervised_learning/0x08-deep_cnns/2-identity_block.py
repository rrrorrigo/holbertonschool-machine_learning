#!/usr/bin/env python3
"""Identity block"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters

    init = K.initializers.HeNormal()

    inputLayer = A_prev
    conv = K.layers.Conv2D(F11, 1,
                           padding='same',
                           kernel_initializer=init)(inputLayer)
    batchNorm = K.layers.BatchNormalization()(conv)
    activation = K.layers.Activation('relu')(batchNorm)
    conv = K.layers.Conv2D(F3, 3,
                           padding='same',
                           kernel_initializer=init)(activation)
    batchNorm = K.layers.BatchNormalization()(conv)
    activation = K.layers.Activation('relu')(batchNorm)
    conv = K.layers.Conv2D(F12, 1,
                           padding='same',
                           kernel_initializer=init)(activation)
    batchNorm = K.layers.BatchNormalization()(conv)
    layer = K.layers.Add()([batchNorm, inputLayer])
    activation = K.layers.Activation('relu')(layer)

    return activation
