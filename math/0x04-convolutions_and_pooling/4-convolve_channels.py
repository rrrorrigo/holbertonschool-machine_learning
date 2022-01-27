#!/usr/bin/env python3
"""3- convolve grayscale"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images with channels

    images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c) containing the
    kernel for the convolution
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
    imagesNum = np.arange(m)
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    if padding == 'same':
        ph = int((ih - 1) * stride[0] + kh - ih // 2 + 1)
        pw = int((iw - 1) * stride[1] + kw - iw // 2 + 1)
    if padding == 'valid':
        ph, pw = (0, 0)
    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    ch = (ih + 2 * ph - kh) // stride[0] + 1
    cw = (iw + 2 * pw - kw) // stride[1] + 1
    output = np.zeros((m, ch, cw))
    for y in range(ch):
        for x in range(cw):
            output[:, x, y] = np.sum(
                kernel * images[:, x * stride[0]: x * stride[0] + kw,
                                y * stride[1]: y * stride[1] + kh],
                axis=(1, 2, 3))
    return np.pad(output, ((0, 0), (ph, ph), (pw, pw)))
