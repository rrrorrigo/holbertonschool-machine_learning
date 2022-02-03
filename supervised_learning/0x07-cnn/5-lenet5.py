#!/usr/bin/env python3
"""1-Input"""


import tensorflow.keras as K


def lenet5(X):
    """Write a function def lenet5(X): that builds a modified version of the
    LeNet-5 architecture using keras:

    X is a K.Input of shape (m, 28, 28, 1) containing the input images for
    the network
        m is the number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method
    All hidden layers requiring activation should use the relu activation
    function
    Returns: a K.Model compiled to use Adam optimization (with default
    hyperparameters) and accuracy metrics"""
    layer1 = K.layers.Conv2D(6, (5, 5), activation='relu', padding='same')(X)
    layer2 = K.layers.AveragePooling2D(2, 2)(layer1)
    layer3 = K.layers.Conv2D(16, (5, 5), activation='relu', padding='valid')(layer2)
    layer4 = K.layers.AveragePooling2D(2, 2)(layer3)
    layer5 = K.layers.Flatten()(layer4)
    layer6 = K.layers.Dense(units=120, activation='relu')(layer5)
    layer7 = K.layers.Dense(units=84, activation='relu')(layer6)
    layer8 = K.layers.Dense(units=10, activation = 'softmax')(layer7)
    model = K.Model(X, layer8)
    opt = K.optimizers.Adam()
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return model
