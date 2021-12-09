#!/usr/bin/env python3
"""Our life is the sum total of all the decisions we make every day,
and those decisions are determined by our priorities"""


def summation_i_squared(n):
    """ that calculates sum{i=1}^{n} i^2:"""
    if n == 1:
        return 1
    return summation_i_squared(n - 1) + n**2