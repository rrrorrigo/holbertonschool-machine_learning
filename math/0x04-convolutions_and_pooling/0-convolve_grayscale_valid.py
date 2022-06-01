#!/usr/bin/env python3
"""Convolve grayscale valid"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ih, iw = (h - kh) + 1, (w - kw) + 1

    cImage = np.zeros(shape=(m, ih, iw))

    for y in range(ih):
        for x in range(iw):
            y0 = y + kh
            x0 = x + kw
            cImage[:, y, x] = np.sum(images[:, y:y0, x:x0] * kernel[...],
                                     axis=(1, 2))

    return cImage
