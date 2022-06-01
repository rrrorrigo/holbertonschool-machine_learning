#!/usr/bin/env python3
"""Convolve image"""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Function that performs a convolution on images using multiple kernels:

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernels is a numpy.ndarray with shape (kh, kw, c, nc) containing the
    kernels for the convolution
        kh is the height of a kernel
        kw is the width of a kernel
        nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
        if ‘same’, performs a same convolution
        if ‘valid’, performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image

    Returns: a numpy.ndarray containing the convolved images"""
    m, h, w, c = images.shape
    kh, kw = kernels.shape[:2]
    sh, sw = stride

    img = images

    if padding == 'same':
        ph, pw = int(((kh - 1) / 2) / sh), int(((kw - 1) / 2) / sw)
        convolveImage = np.zeros(shape=images.shape)
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)
    if padding == 'valid':
        h, w = int((h - kh) / sh + 1), int((w - kw) / sw + 1)
        convolveImage = np.zeros(shape=(m, h, w, c))
    if type(padding) is tuple:
        ph, pw = padding
        convolveImage = np.zeros(shape=images.shape)
        img = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)

    for z in range(c):
        for y in range(h):
            for x in range(w):
                y0 = y * sh
                y1 = y0 + kh
                x0 = x * sw
                x1 = x0 + kw
                convolveImage[:, y, x, z] = np.sum(img[:, y0:y1, x0:x1] * kernels[..., z],
                                                axis=(1, 2, 3))
    return convolveImage
