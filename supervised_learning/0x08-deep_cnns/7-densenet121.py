#!/usr/bin/env python3
"""DenseNet-121 - Densely Connected Convolutional Networks"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture

    growth_rate is the growth rate
    compression is the compression factor
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization and a rectified
    linear activation (ReLU), respectively
    All weights should use he normal initialization
    You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    init = K.initializers.HeNormal()
    inputLayer = K.Input(shape=(224, 224, 3))
    batchNorm = K.layers.BatchNormalization()(inputLayer)
    activation = K.layers.Activation('relu')(batchNorm)
    convolution = K.layers.Conv2D(64, kernel_size=(7, 7),
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=init)(activation)
    maxPooling = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                        padding='same')(convolution)
    denseBlock1 = dense_block(maxPooling, 64, growth_rate, 6)
    transitionBlack1 = transition_layer(denseBlock1[0],
                                        denseBlock1[1],
                                        compression)
    denseBlock2 = dense_block(transitionBlack1[0],
                              transitionBlack1[1],
                              growth_rate, 12)
    transitionBlack2 = transition_layer(denseBlock2[0],
                                        denseBlock2[1],
                                        compression)
    denseBlock3 = dense_block(transitionBlack2[0],
                              transitionBlack2[1],
                              growth_rate, 24)
    transitionBlack3 = transition_layer(denseBlock3[0],
                                        denseBlock3[1],
                                        compression)
    denseBlock4 = dense_block(transitionBlack3[0],
                              transitionBlack3[1],
                              growth_rate, 16)
    avgPooling = K.layers.AveragePooling2D(pool_size=(7, 7),
                                           strides=1)(denseBlock4[0])
    outputLayer = K.layers.Dense(1000,
                                 activation='softmax')(avgPooling)
    model = K.Model(inputLayer, outputLayer)

    return model
