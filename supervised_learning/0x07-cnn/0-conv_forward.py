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
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # output size depends on filter size and must be equal to image size
        # imposing constraints on padding for a given set of strides
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    if padding == 'same':
        # pad A_prev before convolution, padding always symmetric here
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    output = np.zeros(shape=(m,
                             int((h_prev - kh + 2 * ph) / sh + 1),
                             int((w_prev - kw + 2 * pw) / sw + 1),
                             c_new))

    for k in range(c_new):
        for i in range(int((h_prev - kh + 2 * ph) / sh + 1)):
            for j in range(int((w_prev - kw + 2 * pw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    k
                ] = np.sum(
                    A_prev[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]
    output = activation(output)
    return output
