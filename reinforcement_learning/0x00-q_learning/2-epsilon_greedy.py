#!/usr/bin/env python3
"""Epsilon Greedy"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """that uses epsilon-greedy to determine the next action:

    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation

    Returns: the next action index"""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
