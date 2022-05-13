#!/usr/bin/env python3


import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation for the network

    loss: is the loss of the network prediction
    alpha: is the learning rate

    Return: an operation that trains the network using gradient descent"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
