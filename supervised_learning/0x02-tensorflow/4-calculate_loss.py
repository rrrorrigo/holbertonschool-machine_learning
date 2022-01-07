#!/usr/bin/env python3
"""0. Placeholders"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
