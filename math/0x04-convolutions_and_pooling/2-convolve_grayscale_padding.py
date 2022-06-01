#!/usr/bin/env python3
"""Padding Convolution grayscale"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a convolution on grayscale images with custom
    padding:

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
    the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
    the image should be padded with 0’s

    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding

    ih, iw = (h + 2 * ph - kh) + 1, (w + 2 * pw - kw) + 1

    convolveImage = np.zeros(shape=(m, ih, iw))

    paddImage = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                       mode='constant',
                       constant_values=0)

    for y in range(h):
        for x in range(w):
            y0 = y + kh
            x0 = x + kw
            convolveImage[:, y, x] = np.sum(paddImage[:, y:y0, x:x0] * kernel,
                                            axis=(1, 2))
    return convolveImage
