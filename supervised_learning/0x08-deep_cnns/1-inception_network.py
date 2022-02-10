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
    init = K.initializers.he_normal()
    i = K.Input(shape=(224, 224, 3))

    layer1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=init,
                              activation='relu')(i)
    layer2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)(layer1)
    layer3R = K.layers.Conv2D(filters=64,
                               kernel_size=1,
                               padding='same',
                               strides=1,
                               kernel_initializer=init,
                               activation='relu')(layer2)
    layer3 = K.layers.Conv2D(filters=192,
                              kernel_size=3,
                              padding='same',
                              strides=1,
                              kernel_initializer=init,
                              activation='relu')(layer3R)
    layer4 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)(layer3)
    layer5 = inception_block(layer4, [64, 96, 128, 16, 32, 32])
    layer6 = inception_block(layer5, [128, 128, 192, 32, 96, 64])
    layer7 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)(layer6)
    layer8 = inception_block(layer7, [192, 96, 208, 16, 48, 64])
    layer9 = inception_block(layer8, [160, 112, 224, 24, 64, 64])
    layer10 = inception_block(layer9, [128, 128, 256, 24, 64, 64])
    layer11 = inception_block(layer10, [112, 144, 288, 32, 64, 64])
    layer12 = inception_block(layer11, [256, 160, 320, 32, 128, 128])
    layer13 = K.layers.MaxPool2D(pool_size=3,
                                  padding='same',
                                  strides=2)(layer12)
    layer14 = inception_block(layer13, [256, 160, 320, 32, 128, 128])
    layer15 = inception_block(layer14, [384, 192, 384, 48, 128, 128])
    layer16 = K.layers.AvgPool2D(pool_size=7,
                                  padding='same',
                                  strides=None)(layer15)

    layer17 = K.layers.Dropout(0.4)(layer16)

    layer18 = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=init,
                              kernel_regularizer=K.regularizers.l2())(layer17)

    model = K.models.Model(inputs=i, outputs=layer18)
    return model
