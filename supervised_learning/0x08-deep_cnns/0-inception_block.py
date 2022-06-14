#!/usr/bin/env python3
"""Inception block"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block:

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution before the
        3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution before the
        5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution after the max
        pooling (Note : The output shape after the max pooling layer is
        )
    All convolutions inside the inception block use a rectified linear
    activation (ReLU)
    Returns: the concatenated output of the inception block"""
    f1 = K.layers.Conv2D(filters[0], (1, 1), activation='relu', padding="same")(A_prev)
    f3r = K.layers.Conv2D(filters[1], (1, 1), activation='relu', padding="same")(A_prev)
    f3 = K.layers.Conv2D(filters[2], (3, 3), activation='relu', padding="same")(f3r)
    f5r = K.layers.Conv2D(filters[3], (1, 1), activation='relu', padding="same")(A_prev)
    f5 = K.layers.Conv2D(filters[4], (5, 5), activation='relu', padding="same")(f5r)
    fmp = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same")(A_prev)
    fpp = K.layers.Conv2D(filters[5], (1, 1), activation='relu', padding="same")(fmp)

    return K.layers.Concatenate()([f1, f3, f5, fpp])
