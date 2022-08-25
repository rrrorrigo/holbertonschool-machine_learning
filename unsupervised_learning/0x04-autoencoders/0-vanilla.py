#!/usr/bin/env python3
"""Vanilla Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each
    hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent
    space representation

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model"""
    encoder_input = keras.Input(shape=(input_dims,))
    decoder_input = keras.Input(shape=(latent_dims,))

    # encoder layers
    for i, lay in enumerate(hidden_layers):
        if i == 0:
            encoder = keras.layers.Dense(lay, activation='relu')(encoder_input)
        else:
            encoder = keras.layers.Dense(lay, activation='relu')(encoder)

    # latent layer
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoder)
    encoder_model = keras.Model(encoder_input, latent)

    # decoder layers
    decoder = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(decoder_input)

    for i, reverse_lay in enumerate(reversed(hidden_layers[:-1])):
        decoder = keras.layers.Dense(reverse_lay, activation='relu')(decoder)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(decoder)
    decoder_model = keras.Model(decoder_input, decoder_output)
    encoder_output = encoder_model(encoder_input)
    decoder_output = decoder_model(encoder_output)
    autoencoder_model = keras.Model(encoder_input, decoder_output)

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
