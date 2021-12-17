#!/usr/bin/env python3
"""0. Initialize Poisson"""


class Poisson:
    """class Poisson that represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor.

        data: is a list of the data to be used to estimate the distribution
        lambtha: is the expected number of occurences in a given time frame
        """
        self.lambtha = float(lambtha)

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
