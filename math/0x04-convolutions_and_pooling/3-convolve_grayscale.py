#!/usr/bin/env python3
"""3- convolve grayscale"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images"""
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    if padding == 'same':
        ph = int((ih - 1) * stride[0] + kh - ih // 2 + 1)
        pw = int((iw - 1) * stride[1] + kw - iw // 2 + 1)
    elif padding == 'valid':
        ph, pw = (0, 0)
    else:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    ch = (ih + 2 * ph - kh) // 2 + 1
    cw = (iw + 2 * pw - kw) // 2 + 1
    output = np.zeros((m, ch, cw))
    for y in range(ch):
        for x in range(cw):
            output[:, x, y] = np.sum(
                kernel * images[:, x * stride[0]: x * stride[0] + kw,
                y * stride[1]: y * stride[1] + kh], axis=(1, 2)
                )
    return np.pad(output, ((0, 0), (ph, ph), (pw, pw)))
