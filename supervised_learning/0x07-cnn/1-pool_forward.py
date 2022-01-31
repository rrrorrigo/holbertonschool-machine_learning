#!/usr/bin/env python3
"""0 conv_forward"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs pooling on images:
    A_prev is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape
    for the pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    Returns: a numpy.ndarray containing the pooled images"""
    m = A_prev.shape[0]
    ih = A_prev.shape[1]
    iw = A_prev.shape[2]
    chnn = A_prev.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    ch = (ih - kh) // stride[0] + 1
    cw = (iw - kw) // stride[1] + 1
    output = np.zeros((m, ch, cw, chnn))
    pooling = np.max if mode == 'max' else np.average
    sh = stride[0]
    sw = stride[1]
    for y in range(ch):
        for x in range(cw):
            output[:, y, x] = pooling(
                A_prev[:, y * sh: y * sh + kh, x * sw: x * sw + kw],
                axis=(1, 2)
            )
    return output