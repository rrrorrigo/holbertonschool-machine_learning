#!/usr/bin/env python3
"""3-one_hot"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Function that converts a label vector into a one-hot matrix

    The last dimension of the one-hot matrix must be the number of classes

    Returns: the one-hot matrix"""
    return K.utils.to_categorical(labels, num_classes=classes)
