from __future__ import print_function, division, absolute_import

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, ClassifierMixin

from BasePredictor import Stump
from Distribution import GaussianThresholdDistribution as StumpDistribution

class RKSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_neurons=1000,
                 regularization=1):
        self.n_neurons = n_neurons
        self.regularization = regularization
        self.base_predictor = Stump()

    def fit(self, X, y):
        # generate one-hot vector matrix
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        m, n = X.shape
        Y = np.zeros((len(y), self.n_classes))
        for r in range(self.n_classes):
            Y[:, r][y == self.classes[r]] = 1
            Y[:, r][y != self.classes[r]] = -1

        # define the sampling distribution
        self.distribution = StumpDistribution(n_dim=n)

        # generate random weights
        self.W = [self.distribution.sample() for _ in range(self.n_neurons)]

        # propagate examples to hidden layer
        Psi = self.base_predictor.eval(self.W, X)

        # define the linear system to solve
        A = np.dot(Psi.T, Psi) + self.regularization * np.eye(self.n_neurons)
        B = np.dot(Psi.T, Y)

        # compute the output weights
        self.output_weights = sp.linalg.cho_solve(sp.linalg.cho_factor(A), B)

    def predict(self, X):
        # propagate examples to hidden layer
        Psi = self.base_predictor.eval(self.W, X)

        # determine most likely class
        return self.classes[np.argmax(np.dot(Psi, self.output_weights), axis=1)]
    
    def mse(self, X, y):
        assert(self.n_classes == 2)
        Y = np.zeros(len(y))
        Y[y == self.classes[0]] = 1
        Y[y != self.classes[1]] = -1

        Psi = self.base_predictor.eval(self.W, X)
        output = np.dot(Psi, self.output_weights)
        mse = np.mean((output - Y)**2)
        return mse
        

        