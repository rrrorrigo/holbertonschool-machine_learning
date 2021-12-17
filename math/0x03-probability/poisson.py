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
        self.π = 3.1415926536
        self.e = 2.7182818285

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """Method that return the factorial result of n"""
        fac = 1
        for number in range(1, n + 1):
            fac = fac * number
        return fac

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        if type(k) is not int:
            k = int(k)
        if k <= 0:
            return 0
        return self.lambtha**k * self.e**-self.lambtha / self.factorial(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        k = int(k)
        if k <= 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
