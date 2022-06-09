#!/usr/bin/env python3
"""convolution backward"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional layer of
    a neural network:

    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the type of
    padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively"""
    images = A_prev
    m, h, w, c = A_prev.shape
    m, ih, iw, ic = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    ph, pw = (0, 0)
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)
    if padding == 'valid':
        h, w = int((h - kh) / sh + 1), int((w - kw) / sw + 1)
        img = images
    if type(padding) is tuple:
        ph, pw = padding
        h, w = (h + 2 * ph - kh) // sh + 1, (w + 2 * pw - kw) // sw + 1
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant',
                     constant_values=0)

    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA_pad = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant',
                    constant_values=0)
    dA_pad = np.pad(dA, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode="constant", constant_values=(0, 0))

    for img in range(m):
        A_img = dA_pad[img]
        dA_img = dA_pad[img]
        for z in range(ic):
            for y in range(ih):
                for x in range(iw):
                    y0 = y * sh
                    y1 = y0 + kh
                    x0 = x * sw
                    x1 = x0 + kw

                    dA_img[x0:x1, y0:y1] += W[..., z] * dZ[img, x, y, z]
                    dW[..., z] += A_img[x0:x1, y0:y1, :] * dZ[img, x, y, z]
        if padding == 'same':
            dA[img, ...] += dA_img[ph:-ph, pw:-pw]
        if padding == 'valid':
            dA[img, ...] += dA_img
    return dA, dW, db
