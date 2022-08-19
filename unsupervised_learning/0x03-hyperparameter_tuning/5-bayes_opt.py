#!/usr/bin/env python3
"""Gaussian process"""


from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class that performs Bayesian optimization on a noiseless 1D
    Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """Class constructor

        f is the black-box function to be optimized
        X_init is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min, max) representing the bounds of the space
        in which to look for the optimal point
        ac_samples is the number of samples that should be analyzed
        during acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of
        the black-box function
        xsi is the exploration-exploitation factor for acquisition
        minimize is a bool determining whether optimization should be
        performed for minimization (True) or maximization (False)"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Function that calculates the next best sample location

        Returns: X_next, EI
            X_next is a numpy.ndarray of shape (1,) representing
            the next best sample point
            EI is a numpy.ndarray of shape (ac_samples,) containing
            the expected improvement of each potential sample"""
        mu, sigma = self.gp.predict(self.X_s)

        sample = np.amin(self.gp.Y) if self.minimize else np.amax(self.gp.Y)

        # class that avoid floating point number error
        with np.errstate(divide='warn'):
            imp = sample - mu if self.minimize else mu - sample - self.xsi
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Function that optimizes the black-box function

        iterations is the maximum number of iterations to perform
        
        Returns: X_opt, Y_opt
            X_opt is a numpy.ndarray of shape (1,) representing
            the optimal point
            Y_opt is a numpy.ndarray of shape (1,) representing
            the optimal function value"""

        X_opt, Y_opt = 0, 0
        for i in range(iterations):
            X_best = self.acquisition()[0]

            if X_best in self.gp.X:
                break
            Y_best = self.f(X_best)
            self.gp.update(X_best, Y_best)
            if Y_best < Y_opt and self.minimize:
                X_opt, Y_opt = X_best, Y_best
            if not self.minimize and Y_best > Y_opt:
                X_opt, Y_opt = X_best, Y_best

        return X_opt, Y_opt
