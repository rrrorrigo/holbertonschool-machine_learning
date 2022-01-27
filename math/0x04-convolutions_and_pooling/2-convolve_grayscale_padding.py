#!/usr/bin/env python3
"""0. Valid Convolution"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a valid convolution on grayscale images

    images: is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel: is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding: is a tuple of (ph, pw)
        ph is the padding for the height of the image
        pw is the padding for the width of the image
        the image should be padded with 0’s

    Returns: a numpy.ndarray containing the convolved images"""
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output = np.zeros((m, (ih - kh + 1), (iw - kw + 1)))
    for y in range(ih - kh + 1):
        for x in range(iw - kw + 1):
            output[:, x, y] = np.sum(
                kernel * images[:, x: x + kw, y: y + kh], axis=(1, 2)
            )
    ph = padding[0]
    pw = padding[1]
    return np.pad(output, ((0, 0), (ph, ph), (pw, pw)))
