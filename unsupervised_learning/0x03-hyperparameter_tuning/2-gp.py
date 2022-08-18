#!/usr/bin/env python3
"""Gaussian process"""


from tkinter.tix import X_REGION
import numpy as np


class GaussianProcess:
    """class that represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Class constructor

        X_init: is a numpy.ndarray of shape (t, 1) representing the inputs
        already sampled with the black-box function
        Y_init: is a numpy.ndarray of shape (t, 1) representing the outputs
        of the black-box function for each input in X_init
            t: is the number of initial samples
        l: is the length parameter for the kernel
        sigma_f: is the standard deviation given to the output of the
        black-box function
        Sets the public instance attributes X, Y, l, and sigma_f
        corresponding to the respective constructor inputs
        Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Function that calculates the covariance kernel matrix between two
        matrices

        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel use the Radial Basis Function (RBF)

        Returns: the covariance kernel matrix as a numpy.ndarray
        of shape (m, n)"""

        # RBF formula: σ²(-0.5 / l²(∑(X1²) + ∑(X2²) - 2(X1X2)))

        sum_X = np.sum(X1 ** 2, 1).reshape(-1, 1)
        sum_X2 = np.sum(X2 ** 2, 1)

        k = sum_X + sum_X2 - 2 * np.dot(X1, X2.T)

        k = self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * k)

        return k

    def predict(self, X_s):
        """Function that predicts the mean and standard deviation of
        points in a Gaussian process

        X_s is a numpy.ndarray of shape (s, 1) containing all of the
        points whose mean and standard deviation should be calculated
            s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for
            each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the variance
            for each point in X_s, respectively"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + 1e-8 * np.eye(len(X_s))
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, np.diag(cov_s)

    def update(self, X_new, Y_new):
        """Function that updates a Gaussian Process:
        X_new is a numpy.ndarray of shape (1,) that represents
        the new sample point
        Y_new is a numpy.ndarray of shape (1,) that represents
        the new sample function value

        Updates the public instance attributes X, Y, and K"""
        self.X = np.append(self.X, X_new).reshape((-1, 1))
        self.Y = np.append(self.Y, Y_new).reshape((-1, 1))
        self.K = self.kernel(self.X, self.X)
