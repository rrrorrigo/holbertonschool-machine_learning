!/usr/bin/env python3
"""3. Mini-Batch"""


import numpy as np
import tensorflow.compat.v1 as tf


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                    batch_size=32, epochs=5, load_path="/tmp/model.ckpt",
                    save_path="/tmp/model.ckpt"):
    """function that trains a loaded neural network model using mini-batch
    gradient descen

    X_train: is a numpy.ndarray of shape (m, 784) containing the training data
    Y_train: is a one-hot numpy.ndarray of shape (m, 10) containing the training labels
    X_valid: is a numpy.ndarray of shape (m, 784) containing the validation data
    Y_valid: is a one-hot numpy.ndarray of shape (m, 10) containing the validation labels
    batch_size: is the number of data points in a batch
    epochs: is the number of times the training should pass through the whole dataset
    load_path: is the path from which to load the model
    save_path: is the path to where the model should be saved after training

    Returns: the path where the model was saved"""
    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
