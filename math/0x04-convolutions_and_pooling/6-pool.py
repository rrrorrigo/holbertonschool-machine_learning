#!/usr/bin/env python3
"""Pool image"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images:

    images is a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape for the
    pooling
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling

    Returns: a numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ih, iw = int(h / kh), int(w / kw)

    pooled_image = np.zeros((m, ih, iw, c))

    func = (lambda x, ax: np.max(x, axis=ax), lambda x, ax: np.average(x, ax))

    pool_mode = func[0] if mode == 'max' else func[1]

    for y in range(ih):
        for x in range(iw):
            y0 = y * sh
            y1 = y0 + kh
            x0 = x * sw
            x1 = x0 + kw
            pooled_image[:, y, x, :] = pool_mode(images[:, y0:y1, x0:x1, :],
                                                 (1, 2))
    return pooled_image
