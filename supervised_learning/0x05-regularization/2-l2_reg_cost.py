#!/usr/bin/env python3
"""2-l2 regularization cost"""


import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """Function that calculates the cost of a neural network with L2
    regularization

    cost: is a tensor containing the cost of the network without L2
    regularization

    Returns: a tensor containing the cost of the network accounting
    for L2 regularization"""
    return cost + tf.losses.get_regularization_losses()