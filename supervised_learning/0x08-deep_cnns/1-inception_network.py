#!/usr/bin/env python3
"""inception_network"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds an inception network as described in Going Deeper
    with Convolutions (2014)
    1. You can assume the input data will have shape (224, 224, 3)
    2. All convolutions inside and outside the inception block should use a
       rectified linear activation (ReLU)
    Returns:    the keras model
    """
    input = K.layers.Input(shape=(224, 224, 3))  # 0

    layer1 = K.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(input)
    layer1 = K.layers.MaxPool2D(3, strides=2, padding='same')(layer1)

    layer2 = K.layers.Conv2D(64, 1, activation='relu')(layer1)
    layer2 = K.layers.Conv2D(192, 3, padding='same', activation='relu')(layer2)
    layer2 = K.layers.MaxPool2D(3, strides=2, padding='same')(layer2)

    layer3 = inception_block(layer2, [64, 96, 128, 16, 32, 32])
    layer3 = inception_block(layer3, [128, 128, 192, 32, 96, 64])
    layer3 = K.layers.MaxPool2D(3, strides=2, padding='same')(layer3)

    layer4 = inception_block(layer3, [192, 96, 208, 16, 48, 64])
    layer4 = inception_block(layer4, [160, 112, 224, 24, 64, 64])
    layer4 = inception_block(layer4, [128, 128, 256, 24, 64, 64])
    layer4 = inception_block(layer4, [112, 144, 288, 32, 64, 64])
    layer4 = inception_block(layer4, [256, 160, 320, 32, 128, 128])
    layer4 = K.layers.MaxPool2D(3, strides=2, padding='same')(layer4)

    layer5 = inception_block(layer4, [256, 160, 320, 32, 128, 128])
    layer5 = inception_block(layer5, [384, 192, 384, 48, 128, 128])

    layer6 = K.layers.AvgPool2D(7, strides=1)(layer5)
    layer6 = K.layers.Dropout(0.4)(layer6)

    output = K.layers.Dense(1000, activation='softmax')(layer6)
    model = K.models.Model(input, output)

    return model
