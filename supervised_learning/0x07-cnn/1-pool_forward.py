#!/usr/bin/env python3
"""Pooling forward"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation over a pooling layer of a
    neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel for
    the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively

    Returns: the output of the pooling layer"""
    images = A_prev
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ih, iw = (h - kh) // sh + 1, (w - kw) // sw + 1

    pooled_image = np.zeros((m, ih, iw, c))

    func = (lambda x, ax: np.max(x, axis=ax),
            lambda x, ax: np.mean(x, axis=ax))

    pool_mode = func[0] if mode == 'max' else func[1]

    for y in range(ih):
        for x in range(iw):
            y0 = y * sh
            y1 = y0 + kh
            x0 = x * sw
            x1 = x0 + kw
            pooled_image[:, y, x] = pool_mode(images[:, y0:y1, x0:x1],
                                              (1, 2))
    return pooled_image
