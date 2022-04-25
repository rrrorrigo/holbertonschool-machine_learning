#!/usr/bin/env python3
"""Poisson"""


class Poisson:
    """Poisson class that represents a poisson distribution"""
    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, lambtha=1.):
        """Class constructor

        data: is a list of the data to be used to estimate the distribution
        lambtha: is the expected number of occurences in a given time frame
        """
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data) / len(data)