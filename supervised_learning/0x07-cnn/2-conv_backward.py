#!/usr/bin/env python3
"""0 conv_forward"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function def conv_backward(dZ, A_prev, W, b, padding="same",
    stride=(1, 1)): that performs back propagation over a convolutional
    layer of a neural network:

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
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases applied to the convolution
    padding is a string that is either same or valid, indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
    sh is the stride for the height
    sw is the stride for the width
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer (dA_prev), the kernels (dW), and the biases (db), respectively"""
    m = dZ.shape[0]
    ih = dZ.shape[1]
    iw = dZ.shape[2]
    chnn = dZ.shape[3]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    ch = (ih - kh) // stride[0] + 1
    cw = (iw - kw) // stride[1] + 1
    if padding == 'valid':
        ph, pw = (0, 0)
    if padding == 'same':
        ph = int(np.ceil(((stride[0] * h_prev) - stride[0] + kh - h_prev) / 2))
        pw = int(np.ceil(((stride[1] * w_prev) - stride[1] + kw - w_prev) / 2))
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    output = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.zeros(shape=b.shape)
    sh = stride[0]
    sw = stride[1]
    for imagesNumber in range(m):
        for n in range(chnn):
            for y in range(ih):
                for x in range(iw):
                    x0 = x * stride[0]
                    y0 = y * stride[1]
                    X = x0 + kh
                    Y = y0 + kw
                    output[imagesNumber, y0: Y, x0: X, :] += dZ[imagesNumber, y, x, n] * W[:, :, :, n]
                    dW[:, :, :, n] += A_prev[imagesNumber, y0: Y, x0: X, :] * dZ[imagesNumber, y, x, n]
    if padding == 'same':
        output = output[
            :,
            ph: output.shape[1] - ph,
            pw: output.shape[2] - pw,
            :
        ]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    return output, dW, db
