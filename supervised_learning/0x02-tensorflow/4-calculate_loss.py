#!/usr/bin/env python3
"""Calculate loss"""


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Function that calculates the cross entropy loss function

    y: is a placeholder for the labels of the input data
    y_pred: is a tensor containing the network’s predictions

    Return: a tensor containing the loss of the prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
