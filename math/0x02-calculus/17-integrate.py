#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial

    poly: is a list of coefficients representing a polynomial
        - the index of the list represents the power of x that the
          coefficient belongs to
        - Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
    C: is an integer representing the integration constant

    Return: a new list of coefficients representing the integral of
    the polynomial"""
    # i == power of x
    result = [C]
    if type(poly) is not list or len(poly) == 0:
        return None
    for i in range(1, len(poly) + 1):
        x = poly[i - 1] / i
        if (x - int(x)) == 0:
            number = int(x)
        else:
            number = x
        result.append(number)
    return result
