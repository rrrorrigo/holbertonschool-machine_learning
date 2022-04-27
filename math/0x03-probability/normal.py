#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Normal class that represents a normal distribution"""
    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor

        data: is a list of the data to be used to estimate the distribution
        mean: is the mean of the distribution
        stddev: is the standard deviation of the distribution
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        else:
            leng = len(data)
            self.mean = sum(data) / leng
            self.stddev = (sum((x - self.mean)**2 for x in data) / leng)**0.5

    def z_score(self, x):
        """Function that calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Function that calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Function that calculates the value of the PDF for a given x-value
        """
        stdd = self.stddev
        firstPart = 1 / (stdd * ((self.π*2)**0.5))
        secondPart = self.e ** (self.z_score(x)**2 * -0.5)
        return firstPart * secondPart

    def erf(self, x):
        """Fuction that calculates the erf of x"""
        firstPart = (2 / (self.π**0.5))
        secondPart = x - (x**3 / 3) + (x**5 / 10) - (x**7 / 42) + (x**9 / 216)
        return firstPart * secondPart

    def cdf(self, x):
        """Function that calculates the value of the CDF for a given x-value
        """
        z_score_modified = (x - self.mean) / (self.stddev * (2**0.5))
        return (1 / 2) * (1 + self.erf(z_score_modified))
