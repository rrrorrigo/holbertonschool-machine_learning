#!/usr/bin/env python3
"""Sequential model"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library:

    nx: is the number of input features to the network
    layers: is a list containing the number of nodes in each layer of the
    network
    activations: is a list containing the activation functions used for each
    layer: of the network
    lambtha: is the L2 regularization parameter
    keep_prob: is the probability that a node will be kept for dropout

    Returns: the keras model"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 input_shape=(nx,),
                                 kernel_regularizer=regularizer))
        if i < len(layers):
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
