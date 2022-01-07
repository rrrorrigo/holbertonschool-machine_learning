#!/usr/bin/env python3
"""0. Placeholders"""


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network

    loss: is the loss of the network’s prediction
    alpha: is the learning rate"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
