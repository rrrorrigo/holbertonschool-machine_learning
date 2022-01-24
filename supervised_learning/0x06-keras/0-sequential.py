#!/usr/bin/env python3
"""0- Sequential"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that build keras model

    nx: is the number of input features to the network
    layers: is a list containing the number of nodes in each layer of the
    network
    activations: is a list containing the activation functions used for each
    layer of the network
    lambtha: is the L2 regularization parameter
    keep_prob: is the probability that a node will be kept for dropout

    Return: Keras model"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(l2=lambtha)
    model.add(K.Input(shape=(nx,)))
    for i, l in enumerate(layers):
        model.add(K.layers.Dense(l, activation=activations[i],
                  kernel_regularizer=regularizer))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model