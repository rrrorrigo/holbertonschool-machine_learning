#!/usr/bin/env python3
"""Vanilla Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder:

    input_dims is an integer containing the dimensions of the model input
    filters is a list containing the number of nodes for each
    hidden layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent
    space representation
    lambtha is the regularization parameter used for L1 regularization
    on the encoded output

    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model"""
    encod_input = keras.Input(shape=input_dims)
    decoder_input = keras.Input(shape=latent_dims)

    # encoder layers
    for i, lay in enumerate(filters):
        if i == 0:
            encoder = keras.layers.Conv2D(lay, kernel_size=(3, 3),
                                          padding='same',
                                          activation='relu')(encod_input)
        else:
            encoder = keras.layers.Conv2D(lay, kernel_size=(3, 3),
                                          padding='same',
                                          activation='relu')(encoder)
        encoder = keras.layers.MaxPool2D((2, 2),
                                         padding='same')(encoder)

    # latent layer
    latent = encoder
    encoder_model = keras.Model(encod_input, latent)

    # decoder layers
    decoder = keras.layers.Conv2D(filters[-1],
                                  kernel_size=(3, 3),
                                  padding='same',
                                  activation='relu')(decoder_input)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)

    for i, reverse_lay in enumerate(reversed(filters[:-2])):
        decoder = keras.layers.Conv2D(reverse_lay,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      activation='relu')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder = keras.layers.Conv2D(filters[0],
                                  kernel_size=(3, 3),
                                  padding='valid',
                                  activation='relu')(decoder)
    decoder = keras.layers.UpSampling2D((2, 2))(decoder)
    decoder_output = keras.layers.Conv2D(input_dims[-1],
                                         kernel_size=(3, 3),
                                         padding='same',
                                         activation='sigmoid')(decoder)
    decoder_model = keras.Model(decoder_input, decoder_output)
    encoder_output = encoder_model(encod_input)
    decoder_output = decoder_model(encoder_output)
    autoencoder_model = keras.Model(encod_input, decoder_output)

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
