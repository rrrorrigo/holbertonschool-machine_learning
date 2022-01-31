#!/usr/bin/env python3
"""0 conv_forward"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs a convolution on images using multiple kernels

    A_prev: is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    W: is a numpy.ndarray with shape (kh, kw, c, nc) containing
    the kernels for the convolution
        kh is the height of a kernel
        kw is the width of a kernel
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output}
    b: is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation: is an activation function applied to the convolution
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    stride: is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images"""
    m = A_prev.shape[0]
    imagesNum = np.arange(m)
    ih = A_prev.shape[1]
    iw = A_prev.shape[2]
    chnn = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    kp = W.shape[2]
    ko = W.shape[3]
    if padding == 'same':
        ph = int((ih - 1) * stride[0] + kh - ih // 2 + 1)
        pw = int((iw - 1) * stride[1] + kw - iw // 2 + 1)
    if padding == 'valid':
        ph, pw = (0, 0)
    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    ch = (ih + 2 * ph - kh) // stride[0] + 1
    cw = (iw + 2 * pw - kw) // stride[1] + 1
    output = np.zeros((m, ch, cw, ko))
    for y in range(ch):
        for x in range(cw):
            for n in range(ko):
                x0 = x * stride[0]
                y0 = y * stride[1]
                X = x0 + kh
                Y = y0 + kw
                output[imagesNum, x, y, n] = activation(np.sum(
                    W[:, :, :, n] * images[imagesNum, x0: X, y0: Y],
                axis=(1, 2, 3)) + b[:, :, :, n])
    return output
