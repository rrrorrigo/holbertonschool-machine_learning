#!/usr/bin/env python3
"""Markov chain"""


import numpy as np


def regular(P):
    """Function that determines the steady state probabilities
    of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) containing the steady
    state probabilities, or None on failure"""
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None

    S = np.full((1, n), (1 / n))
    new_P = np.copy(P)
    Sx = S

    while True:
        new_P = np.matmul(new_P, P)

        if np.any(new_P <= 0):
            return None

        S = np.matmul(S, P)
        if np.all(S == Sx):
            return S
        Sx = S
