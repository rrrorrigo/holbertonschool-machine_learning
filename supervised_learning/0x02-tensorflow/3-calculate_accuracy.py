#!/usr/bin/env python3
"""Calculate accuracy"""


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction

    y: is a placeholder for the labels of the inpu data
    y_pred: is a tensor containing the network's predictions

    Returns: a tensor containing the decimal accuracy of the prediction"""
    return tf.metrics.accuracy(y, y_pred)[0]
