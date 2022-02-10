#!/usr/bin/env python3
"""
    Projection Block - Deep Residual Learning for Image Recognition
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block as described in Deep Residual
    Learning for Image Recognition (2015):

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution as well
        as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main path and the
    shortcut connection
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the activated output of the projection block
    """
    F11, F3, F12 = filters

    init = K.initializers.HeNormal()
    inputLayer = A_prev
    conv = K.layers.Conv2D(F11, 1,
                            padding='same',
                            strides=s,
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

    conv = K.layers.Conv2D(F12, 1,
                              padding='same',
                              strides=s,
                              kernel_initializer=init)(inputLayer)
    batchNorm = K.layers.BatchNormalization()(conv)
    add = K.layers.Add()([batchNorm, batchNorm])
    activation = K.layers.Activation('relu')(add)

    return activation