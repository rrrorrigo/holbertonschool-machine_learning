#!/usr/bin/env python3
"""Convoltion forward"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation over a convolutional layer
    of a neural network:

    A_prev: is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m: is the number of examples
        h_prev: is the height of the previous layer
        w_prev: is the width of the previous layer
        c_prev: is the number of channels in the previous layer
    W: is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh: is the filter height
        kw: is the filter width
        c_prev: is the number of channels in the previous layer
        c_new: is the number of channels in the output
    b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation: is an activation function applied to the convolution
    padding: is a string that is either same or valid, indicating the type of
    padding used
    stride: is a tuple of (sh, sw) containing the strides for the convolution
        sh: is the stride for the height
        sw: is the stride for the width

    Returns: the output of the convolutional layer"""
    images = A_prev
    m, h, w, c = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
        output = np.zeros((m, h, w, c_new))
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)
    if padding == 'valid':
        h, w = int((h - kh) / sh + 1), int((w - kw) / sw + 1)
        img = images
        output = np.zeros((m, h, w, c_new))
    if type(padding) is tuple:
        ph, pw = padding
        h, w = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        output = np.zeros((m, h, w, c_new))
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)

    for z in range(c_new):
        for y in range(h):
            for x in range(w):
                y0 = y * sh
                y1 = y0 + kh
                x0 = x * sw
                x1 = x0 + kw
                result = np.sum(img[:, y0:y1, x0:x1, :] * W[..., z],
                                axis=(1, 2, 3))
                output[:, y, x, z] = result
    return activation(output + b)
