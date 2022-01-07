#!/usr/bin/env python3
"""0. Placeholders"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction

    y is a: placeholder for the labels of the input data
    y_pred: is a tensor containing the network’s predictions

    Returns: a tensor containing the decimal accuracy of the prediction"""
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    accurrancy = tf.reduce_mean(tf.cast(tf.math.equal(y_pred, y), "float"))
    return accurrancy
