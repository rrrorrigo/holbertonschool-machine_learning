#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block as described in
    https://arxiv.org/pdf/1409.4842.pdf

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution before
        the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution before
        the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution after the
        max pooling (Note : The output shape after the max pooling layer
        is outputshape = math.floor((inputshape - 1) / strides) + 1)
    All convolutions inside the inception block should use a rectified
    linear activation (ReLU)

    Returns: the concatenated output of the inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    layer1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    layer2 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    layer2 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(layer2)

    layer3 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    layer3 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(layer3)

    layer4 = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    layer4 = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(layer4)

    output = K.layers.Concatenate()([layer1, layer2, layer3, layer4])
    return output