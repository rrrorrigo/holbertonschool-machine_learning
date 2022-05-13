#!/usr/bin/env python3
"""train"""


import tensorflow.compat.v1 as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """Function that builds, trains, and saves a neural network classifier

    X_train: is a numpy.ndarray containing the training input data
    Y_train: is a numpy.ndarray containing the training labels
    X_valid: is a numpy.ndarray containing the validation input data
    Y_valid: is a numpy.ndarray containing the validation labels
    layer_sizes: is a list containing the number of nodes in each layer of the
        network
    activations: is a list containing the activation functions for each layer
        of the network
    alpha: is the learning rate
    iterations: is the number of iterations to train over
    save_path: designates where to save the model

    Return: the path where the model was saved"""
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    createPH = __import__('0-create_placeholders').create_placeholders
    create_train_op = __import__('5-create_train_op').create_train_op
    forward_prop = __import__('2-forward_prop').forward_prop

    x, y = createPH(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection("y_pred", y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    training = create_train_op(loss, alpha)
    tf.add_to_collection("train", train)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost))
                print("\tTraining Accuracy: {}".format(accuracy_train))
                print("\tValidation Cost: {}".format(cost_val))
                print("\tValidation Accuracy: {}".format(accuracy_val))
            if i < iterations:
                sess.run(training, feed_dict={x: X_train, y: Y_train})
        return saver.save(sess, save_path)