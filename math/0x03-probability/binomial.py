#!/usr/bin/env python3
"""10. Initialize Binomial"""


class Binomial:
    """class Binomial that represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Class contructor

        data: is a list of the data to be used to estimate the distribution
        n: is the number of Bernoulli trials
        p: is the probability of a “success”"""
        self.n = n
        self.p = p

        self.π = 3.1415926536
        self.e = 2.7182818285

        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            num = 0
            lambtha = sum(data) / len(data)
            for x in data:
                num += (x - lambtha) ** 2
            x = num / len(data)
            self.p = 1 - (x / lambtha)
            self.n = round(lambtha / self.p)
            self.p = lambtha / self.n

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)

        if k < 0:
            return 0
        nFact = 1
        xFact = 1
        opFact = 1
        for i in range(1, self.n + 1):
            nFact *= i
        for i in range(1, k + 1):
            xFact *= i
        for i in range(1, (self.n - k) + 1):
            opFact *= i
        a = ((nFact) / (xFact * opFact))
        return (a * (self.p**k) * ((1 - self.p)**(self.n - k)))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”"""
        k = int(k)

        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
