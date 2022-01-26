#!/usr/bin/env python3
"""0. Valid Convolution"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    Returns: a numpy.ndarray containing the convolved images"""
    m = images.shape[0]
    imagesNum = np.arange(m)
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output = np.zeros((m, (ih - kh + 1), (iw - kw + 1)))
    for y in range(ih - kh + 1):
        for x in range(iw - kw + 1):
            output[imagesNum, x, y] = np.sum(
                kernel * images[imagesNum, x: x + kw, y: y + kh], axis=(1, 2)
            )
    oh = output.shape[1]
    ow = output.shape[2]
    diffHeight = int((ih - oh) / 2)
    diffWidth = int((iw - ow) / 2)
    return np.pad(
        output, ((0, 0), (diffHeight, diffHeight), (diffWidth, diffWidth))
    )
