#!/usr/bin/env python3
"""6. Initialize Normal"""


class Normal:
    """class Normal that represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Class contructor

        data: is a list of the data to be used to estimate the distribution
        mean: is the mean of the distribution
        stddev: is the standard deviation of the distribution"""
        self.mean = float(mean)
        self.stddev = float(stddev)

        self.π = 3.1415926536
        self.e = 2.7182818285

        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        else:
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        a = (1 / (self.stddev * ((2 * self.π) ** (1 / 2))))
        b = (-1 / 2) * ((x - self.mean) / self.stddev) ** 2
        return a * (self.e ** b)

    def erf(self, x):
        """Error function of x"""
        erf = 2 / (self.π**0.5) * (x - (x**3) / 3 + (x ** 5) / 10 -
            (x ** 7) / 42 + (x ** 9) / 216)
        return erf

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        x = (x - self.mean) / (self.stddev * (2**0.5))
        erf = self.erf(x)
        return (1 + erf) / 2
