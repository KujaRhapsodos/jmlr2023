import math
import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from scipy.stats import norm


class Distribution:
    def __init__(self, n_dim, rng=None) -> None:
        self.n_dim = n_dim
        self.rng = default_rng(rng)

    def sample(self, n):
        raise NotImplementedError

    def score_samples(self, x):
        raise NotImplementedError


class GaussianDistribution(Distribution):
    def __init__(self, n_dim, mean=None, sigma=1.0, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.sigma = sigma
        if mean is None:
            self.mean = np.array([0]*self.n_dim)
        else:
            self.mean = mean

    def sample(self):
        return self.rng.normal(self.mean, self.sigma)

    def score_samples(self, x):
        rv = multivariate_normal(mean=self.mean, cov=self.sigma**2)
        return rv.logpdf(x)

    def set_mean(self, mean):
        self.mean = mean


class GaussianThresholdDistribution(Distribution):
    def __init__(self, n_dim, sigma=1.0, rng=None) -> None:
        super().__init__(n_dim, rng)
        self.sigma = sigma

    def sample(self):
        idx = self.rng.integers(self.n_dim)
        param = self.rng.normal(0, self.sigma)
        return (idx, param)
    
    def score_samples(self, x):
        rv = norm(loc=0, scale=self.sigma)
        return [math.log(1 / self.n_dim) + rv.logpdf(z) for (_, z) in x]


    