import numpy as np
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    init = tf.keras.initializers.VarianceScaling(scale=2.0)
    layer1 = tf.layers.Conv2D(6, (5, 5), kernel_regularizer=init, padding='same', activation=tf.nn.relu())(x)
    layer2 = tf.layers.MaxPooling2D(pool_size=2, kernel_regularizer=init, strides=2, activation=tf.nn.relu())(layer1)
    layer3 = tf.layers.Conv2D(16, (5, 5), kernel_regularizer=init, padding='valid', activation=tf.nn.relu())(layer2)
    layer4 = tf.layers.MaxPooling2D(pool_size=2, strides=2, kernel_regularizer=init, activation=tf.nn.relu())(layer3)
    layer5 = tf.layers.Dense(120, kernel_regularizer=init, activation=tf.nn.relu())(layer4)
    layer6 = tf.layers.Dense(84, kernel_regularizer=init, activation=tf.nn.relu())(layer5)
    layer7 = tf.layers.Dense(10, kernel_regularizer=init, activation=None)(layer6)

    activation = tf.nn.softmax(layer7)
    losses = tf.losses.softmax_cross_entropy(y, layer7)
    