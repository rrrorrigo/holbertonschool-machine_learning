#!/usr/bin/env python3
"""Calculate accuracy"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction

    y: is a placeholder for the labels of the input data
    y_pred: is a tensor containing the network's predictions

    Returns: a tensor containing the decimal accuracy of the prediction"""
    equality = tf.math.equal(y_pred, y)
    acc = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return acc
