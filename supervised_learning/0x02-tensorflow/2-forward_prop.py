#!/usr/bin/env python3
"""Forward propagation"""


import tensorflow.compat.v1 as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation graph
    for the neural network

    x: is the placeholder for the input data
    layer_sizes: is a list containing the number of nodes in each layer of
    the network
    activations: is a list containing the activation functions for each layer
    of the network

    Return: the prediction of the network in tensor form"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            prev = create_layer(x, layer_sizes[i], activations[i])
        else:
            prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
