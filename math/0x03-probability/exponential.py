#!/usr/bin/env python3
"""3. Initialize Exponential"""


class Exponential:
    """class Exponential that represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Class contructor

        data: is a list of the data to be used to estimate the distribution
        lambtha: is the expected number of occurences in a given time frame
        """
        self.lambtha = float(lambtha)
        self.π = 3.1415926536
        self.e = 2.7182818285
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def factorial(self, n):
        """Method that return the factorial result of n"""
        fac = 1
        for number in range(1, n + 1):
            fac = fac * number
        return fac

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""
        if x < 0:
            return 0
        return self.lambtha * self.e ** (-1 * self.lambtha * x)

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""
        if x < 0:
            return 0
        return 1 - self.e ** (-1 * self.lambtha * x)
