#!/usr/bin/env python3
"""0. Placeholders"""


import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5 architecture
    using tensorflow

    x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
    images for the network
        m is the number of images
    y is a tf.placeholder of shape (m, 10) containing the one-hot labels
    for the network
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid
            padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method: tf.keras.initializers.VarianceScaling
    (scale=2.0)
    All hidden layers requiring activation should use the relu
    activation function
    you may import tensorflow.compat.v1 as tf
    you may NOT use tf.keras only for the he_normal method.
    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
        (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    layer1 = tf.layers.Conv2D(6, (5, 5), kernel_initializer=init, padding='same', activation=tf.nn.relu)(x)
    layer2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer1)
    layer3 = tf.layers.Conv2D(16, (5, 5), kernel_initializer=init, padding='valid', activation=tf.nn.relu)(layer2)
    layer4 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(layer3)
    layer5 = tf.layers.Flatten()(layer4)
    layer6 = tf.layers.Dense(120, kernel_initializer=init, activation=tf.nn.relu)(layer5)
    layer7 = tf.layers.Dense(84, kernel_initializer=init, activation=tf.nn.relu)(layer6)
    layer8 = tf.layers.Dense(10, kernel_initializer=init)(layer7)

    activation = tf.nn.softmax(layer8)
    print(y.shape, layer7.shape)
    losses = tf.losses.softmax_cross_entropy(y, layer8)
    accurrancy = tf.reduce_mean(tf.cast(tf.math.equal(tf.argmax(layer8, 1), tf.argmax(y, 1)), "float"))
    adam = tf.train.AdamOptimizer().minimize(losses)

    return activation, adam, losses, accurrancy
    