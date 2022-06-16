#!/usr/bin/env python3
"""Inception Network"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """that builds the inception network as described in Going Deeper with
    Convolutions (2014):

    the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block use a
    rectified linear activation (ReLU)

    Returns: the keras model"""
    init = K.initializers.he_normal()
    x = K.Input(shape=(224, 224, 3))
    layer1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                             kernel_initializer=init, activation='relu')(x)
    layer2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding='same')(layer1)
    layer3 = K.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                             kernel_initializer=init,
                             activation='relu')(layer2)
    layer4 = K.layers.Conv2D(192, (3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=init,
                             activation='relu')(layer3)
    layer4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding='same')(layer4)

    # Inception 3a
    Y = [64, 96, 128, 16, 32, 32]
    layer5 = inception_block(layer4, Y)

    # Inception 3b
    Y = [128, 128, 192, 32, 96, 64]
    layer7 = inception_block(layer5, Y)

    layer9 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding='same')(layer7)

    # Inception 4a
    Y = [192, 96, 208, 16, 48, 64]
    layer10 = inception_block(layer9, Y)

    # Inception 4b
    Y = [160, 112, 224, 24, 64, 64]
    layer12 = inception_block(layer10, Y)

    # Inception 4c
    Y = [128, 128, 256, 24, 64, 64]
    layer14 = inception_block(layer12, Y)

    # Inception 4d
    Y = [112, 144, 288, 32, 64, 64]
    layer16 = inception_block(layer14, Y)

    # Inception 4e
    Y = [256, 160, 320, 32, 128, 128]
    layer18 = inception_block(layer16, Y)

    layer20 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                    padding='same')(layer18)

    # Inception 5a
    Y = [256, 160, 320, 32, 128, 128]
    layer21 = inception_block(layer20, Y)

    # Inception 5b
    Y = [384, 192, 384, 48, 128, 128]
    layer23 = inception_block(layer21, Y)

    layer25 = K.layers.AveragePooling2D((7, 7), strides=1)(layer23)

    layer26 = K.layers.Dropout(.40)(layer25)

    output = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(layer26)

    model = K.Model(x, outputs=output)

    return model
