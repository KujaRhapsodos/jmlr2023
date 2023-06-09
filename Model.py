from __future__ import annotations

from copy import deepcopy
import math
import matplotlib.pyplot as plot
import numpy as np
import scipy as sp
import seaborn
from sklearn import mixture ; seaborn.set()
from scipy.special import erf, binom
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso

from BasePredictor import *
from Distribution import *
from Kernel import *
from Loss import *
from WeightFunction import WeightFunction

def array_plus_vector(arr: np.ndarray, vec: np.ndarray) -> np.ndarray:
    m, n = arr.shape
    vec = vec.flatten()
    if vec.shape[0] == n:
        return arr + vec
    else:
        assert(vec.shape[0] == m)
        return (arr.T + vec).T

def array_times_vector(arr: np.ndarray, vec: np.ndarray, axis) -> np.ndarray:
    m, n = arr.shape
    vec = vec.flatten()
    if axis == 1:
        assert(vec.shape[0] == n)
        return np.multiply(arr, vec)
    elif axis == 0:
        assert(vec.shape[0] == m)
        return (arr.T * vec).T

def array_times_array(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    assert(arr1.shape == arr2.shape)
    return np.multiply(arr1, arr2)

def get_n_dim_from_input(input) -> int:
    if type(input) is int:
        n_dim = input
    elif type(input) is np.ndarray:
        n_dim = input.shape[1]
    else:
        print("Invalid 'input' : {}".format(input))
    return n_dim

class Model(WeightFunction):

    def __init__(self, dist: Distribution, kernel: Kernel, base_pred: BasePredictor, rng=None,
                 max_n_mc=10000, n_mc=None, use_mc=True, mc_precision=0.01) -> None:
        self.dist = dist
        self.base_pred = base_pred
        self.rng = np.random.default_rng(rng)
        self.max_n_mc = max_n_mc
        self.n_mc = n_mc
        self.use_mc = use_mc
        self.mc_precision = mc_precision
        super().__init__(kernel=kernel)
    
    def expectations(self, X) -> np.ndarray:
        """
        Returns the (n_examples, n_centers) array containing
        the value of the expectation for each (center, x) pair.
        """
        if self.use_mc:
            expectations = self._mc_expectations(X, self.n_mc)
        else:
            expectations = self._exact_expectations(X)
        return expectations

    def _exact_expectations(self, X) -> np.ndarray:
        """
        This function should ideally be implemented
        for any new instantiation. Otherwise, Monte Carlo
        will be used, which is slower and less accurate.

        Use self.set_use_mc(False) to use this function.
        """
        raise NotImplementedError
    
    def _mc_expectations(self, X: np.ndarray, n_mc=None) -> np.ndarray:
        if n_mc is None:
            n_mc = self.get_expect_n_mc()
            print("Got n mc")
        list_of_w = [self.sample_center() for _ in range(n_mc)]
        base_predictions = self.base_pred.eval(list_of_w, X)       # m x N
        gram = self.kernel.calculate(list_of_w, self.list_of_centers) # N x T
        return (1 / n_mc) * np.dot(base_predictions, gram)

    def get_expect_n_mc(self):
        if self.n_mc is None:
            target_var = self.mc_precision**2
            max_center_norm = self.get_max_center_norm()
            n_mc = int(min(max_center_norm**2 * self.kappa()**2 / target_var, self.max_n_mc))
        else:
            n_mc = self.n_mc
        return n_mc

    def get_empty_expectations(self, X):
        m = X.shape[0]
        T = self.get_n_centers()
        return np.zeros(shape=(m, T))

    def get_scalar_over_norm(self, X):
        W = np.array(self.list_of_centers)
        return self._get_scalar_over_norm(X, W)
    
    def _get_scalar_over_norm(self, X, W):
        X_norms = np.linalg.norm(X, axis=1)
        X_norms[X_norms == 0] = 1
        return array_times_vector(np.dot(W, X.T), 1.0 / X_norms, axis=1).T
    
    def output(self, X: np.ndarray) -> np.ndarray:
        if self.get_n_centers() == 0:
            output = np.zeros(shape=X.shape[0])
        else:
            if self.use_mc:
                output = self.mc_output(X)
            else:
                output = np.dot(self.expectations(X), self.coefs).flatten()
        return output
    
    def mc_output(self, X: np.ndarray, n_mc=None) -> np.ndarray:
        if n_mc is None:
            n_mc = self.get_output_n_mc()
        list_of_w = [self.sample_center() for _ in range(n_mc)]
        base_predictions = self.base_pred.eval(list_of_w, X)
        weight_func_values = self.eval_weight_func_multiple_centers(list_of_w)
        return (1 / n_mc) * np.dot(base_predictions, weight_func_values)

    def get_output_n_mc(self):
        if self.n_mc is None:
            target_var = self.mc_precision**2
            n_mc = int(self.norm()**2 * self.kappa()**2 / target_var)
            n_mc = int(min(n_mc, self.max_n_mc))
        else:
            n_mc = self.n_mc
        return n_mc

    def efficient_update(self, center, coef, scale=1.0):
        """
        Multiplies the weight function by scale, then add
        the new center and coefficient.

        The scale is useful when regularizing.
        """
        self *= scale
        return self.add_center(center, coef)

    def theta(self):
        return self.iota()
    
    def mc_theta(self, X: np.ndarray) -> float:
        m = X.shape[0]
        T = self.max_n_mc
        centers_left = [self.sample_center() for _ in range(T)]
        centers_right = [self.sample_center() for _ in range(T)]
        kern_diag = self.kernel.diag(centers_left, centers_right)
        nonzero_idx = np.flatnonzero(kern_diag)
        print(len(nonzero_idx))
        if len(nonzero_idx) == 0:
            return 0
        base_pred_left = np.zeros((m, T))
        base_pred_right = np.zeros((m, T))
        base_pred_left[:, nonzero_idx] = self.base_pred.eval([centers_left[i] for i in nonzero_idx], X)
        base_pred_right[:, nonzero_idx] = self.base_pred.eval([centers_right[i] for i in nonzero_idx], X)
        candidates_squared = np.mean(base_pred_left * base_pred_right * kern_diag, axis=1)
        return float(np.sqrt(max(candidates_squared)))

    def iota(self):
        return self.kappa()
    
    def mc_iota(self, X: np.ndarray) -> float:
        m = X.shape[0]
        T = self.max_n_mc
        centers = [self.sample_center() for _ in range(T)]
        kern_diag = self.kernel.diag(centers, centers)
        nonzero_idx = np.flatnonzero(kern_diag)
        if len(nonzero_idx) == 0:
            return 0
        base_predictions = np.zeros((m, T))
        nonzero_centers = [centers[i] for i in nonzero_idx]
        base_predictions[:, nonzero_idx] = self.base_pred.eval(nonzero_centers, X) # m x T
        rdv = np.sqrt(base_predictions**2 * kern_diag)
        candidates = np.mean(rdv, axis=1)
        return float(max(candidates))

    def kappa(self):
        raise NotImplementedError
    
    def mc_kappa(self, X: np.ndarray) -> float:
        m = X.shape[0]
        T = self.max_n_mc
        centers = [self.sample_center() for _ in range(self.max_n_mc)]
        kern_diag = self.kernel.diag(centers, centers)
        nonzero_idx = np.flatnonzero(kern_diag)
        if len(nonzero_idx) == 0:
            return 0
        base_predictions = np.zeros((m, T))
        nonzero_centers = [centers[i] for i in nonzero_idx]
        base_predictions[:, nonzero_idx] = self.base_pred.eval(nonzero_centers, X) # m x T
        rdv = base_predictions**2 * kern_diag
        candidates_squared = np.mean(rdv, axis=1)
        return float(np.sqrt(max(candidates_squared)))
    
    def operator_norm(self):
        return min(self.theta(), self.iota(), self.kappa())

    def get_n_dim(self):
        return self.dist.n_dim

    def sample_center(self):
        return self.dist.sample()

    def max_output(self):
        return self.norm() * self.operator_norm()

    def eval_base_predictor(self, W, X):
        return self.base_pred.eval(W, X)

    def set_use_mc(self, boolean: bool):
        self.use_mc = boolean

    def set_mc_precision(self, precision: float):
        self.mc_precision = precision

    def set_max_n_mc(self, max_n: int):
        self.max_n_mc = max_n

    def set_n_mc(self, n_mc: int):
        self.n_mc = n_mc

    def copy(self) -> Model:
        return deepcopy(self)

class Instantiation1(Model):
    def __init__(self, input, sigma='auto', gamma='auto', target_variance='auto', 
                 rng=None, use_mc=False, **kwargs) -> None:
        n_dim = get_n_dim_from_input(input)
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma, target_variance, n_dim=n_dim)
        dist = GaussianDistribution(n_dim=n_dim, sigma=sigma, rng=rng)
        kernel = GaussianKernel(gamma=gamma)
        base = Sign()
        super().__init__(dist, kernel, base, rng, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X) -> np.ndarray:
        n = self.get_n_dim()
        w_0 = self.dist.mean
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2
        sqrt2zeta = 1 / math.sqrt(0.5 * (1/s**2 + 1/g**2))
        W = np.array(self.list_of_centers)
        W_prime = (s**2 * W + g**2 * w_0) / s2g2
        W_norms = np.linalg.norm(W, axis=1)
        w_0_norm = np.sqrt(np.dot(w_0, w_0))
        mixed_norms = np.linalg.norm(W / g**2 + w_0 / s**2, axis=1)
        exp_norms = np.exp(-W_norms**2 / (2*g**2) \
                           -w_0_norm**2 / (2*s**2) \
                           +(s**2 * g**2 / (2*s2g2)) * mixed_norms**2)        
        arr = self._get_scalar_over_norm(X, W_prime)
        global_coef = (1 + s**2 / g**2)**(-n/2)  

        return global_coef * array_times_vector(erf(arr / sqrt2zeta), exp_norms, axis=1)
    
    def _exact_centered_expectations(self, X) -> np.ndarray:
        n = self.get_n_dim()
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2

        W = np.array(self.list_of_centers)
        W_norms = np.linalg.norm(W, axis=1)
        exp_norms = np.exp(-W_norms**2 / (2*s2g2))

        arr = self.get_scalar_over_norm(X)
        erf_coef = 1 / (g * math.sqrt(2 * (1 + g**2 / s**2)))
        global_coef = (1 + s**2 / g**2)**(-n/2)  

        return global_coef * array_times_vector(erf(erf_coef * arr), exp_norms, axis=1)

    def theta(self):
        s = self.sigma
        g = self.gamma
        n = self.get_n_dim()
        return (1 + 2*s**2 / g**2)**(-n/4)

    def iota(self):
        return 1.0

    def kappa(self):
        return 1.0

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)
            
    def get_adjusted_gamma(self, gamma, target_variance, n_dim):
        if gamma == 'auto':
            var = self.get_adjusted_target_var(target_variance)
            return self.get_gamma_from_theta(var, n_dim)
        else:
            return float(gamma)

    def get_adjusted_target_var(self, target_variance):
        return 0.5 if target_variance == 'auto' else float(target_variance)

    def get_gamma_from_theta(self, theta, n_dim):
        return math.sqrt(2) * self.sigma / math.sqrt(theta**(-4/n_dim) - 1)



class Instantiation2(Model):
    def __init__(self, input, sigma='auto', gamma='auto', rng=None, use_mc=False, **kwargs) -> None:
        n_dim = get_n_dim_from_input(input)
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma)
        dist = GaussianThresholdDistribution(n_dim=n_dim, sigma=sigma, rng=rng)
        kernel = IndicatorGaussianKernel(gamma=gamma)
        base = Stump()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X) -> np.ndarray:
        n = self.get_n_dim()
        s = self.sigma
        g = self.gamma
        s2g2 = s**2 + g**2

        W = np.array(self.list_of_centers)
        W_indices = W[:, 0].astype(int)
        W_values = W[:, 1]

        zeta = math.sqrt(1 / (1/s**2 + 1/g**2))
        sqrt2zeta = math.sqrt(2) * zeta
        W2prime = s**2 / s2g2 * W_values

        coef = zeta / (s * n)
        exp_norms = np.exp(-W_values**2 / (2 * s2g2))
        erf_stump = erf((X[:,W_indices]-W2prime)/sqrt2zeta)

        return coef * array_times_vector(erf_stump, exp_norms, axis=1)

    def theta(self):
        s = self.sigma
        g = self.gamma
        n = self.get_n_dim()
        return (1 + 2*s**2 / g**2)**(-1/4) / math.sqrt(n)

    def iota(self):
        return 1.0

    def kappa(self):
        return 1.0

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)
            
    def get_adjusted_gamma(self, gamma):
        return self.sigma if gamma == 'auto' else float(gamma)
        


    """
    Exponential kernel, gaussian distribution, sign
    
    Parameters
    ----------

    input : int or np.ndarray,
        Number of input features, or data array of shape (n_examples, n_features).
    """
    def __init__(self, input, sigma='auto', gamma='auto', target_variance='auto', rng=None, use_mc=False, **kwargs) -> None:
        n_dim = get_n_dim_from_input(input)
        sigma = self.sigma = self.get_adjusted_sigma(sigma)
        gamma = self.gamma = self.get_adjusted_gamma(gamma, target_variance, n_dim)
        dist = GaussianDistribution(n_dim=n_dim, sigma=sigma, rng=rng)
        kernel = ExponentialKernel(gamma=gamma)
        base = Sign()
        super().__init__(dist, kernel, base, rng=None, use_mc=use_mc, **kwargs)
    
    def _exact_expectations(self, X) -> np.ndarray:
        s = self.sigma
        g = self.gamma

        W = np.array(self.list_of_centers)
        W_norms = np.linalg.norm(W, axis=1)

        cst = s / (g**2 * math.sqrt(8 * math.pi))
        exp_norms = np.exp((cst * W_norms) ** 2)
        arr = self.get_scalar_over_norm(X)

        return array_times_vector(erf(cst * arr), exp_norms, axis=1)

    def theta(self):
        s = self.sigma
        g = self.gamma
        n = self.get_n_dim()
        return (1 - s**2/ (2*g**2))**(-n/2)

    def iota(self):
        s = self.sigma
        g = self.gamma
        n = self.get_n_dim()
        return (1 - s**2 / (2*g**2))**(-n/2)

    def kappa(self):
        s = self.sigma
        g = self.gamma
        n = self.get_n_dim()
        return (1 - s**2 / g**2)**(-n/4)

    def get_adjusted_sigma(self, sigma):
        return 1.0 if sigma == 'auto' else float(sigma)

    def get_adjusted_gamma(self, gamma, target_variance, n_dim):
        if gamma == 'auto':
            var = self.get_adjusted_target_var(target_variance)
            return self.get_gamma_from_theta(var, n_dim)
            return self.get_gamma_old_formula(var, n_dim)
        else:
            return float(gamma)

    def get_adjusted_target_var(self, target_variance):
        return 1.5 if target_variance == 'auto' else float(target_variance)

    def get_gamma_from_theta(self, theta, n_dim):
        return math.sqrt(0.5) * self.sigma / math.sqrt(1 - theta**(-2/n_dim))

    def get_gamma_old_formula(self, theta, n_dim):
        return math.sqrt(2) * self.sigma / math.sqrt(theta**(-4/n_dim) - 1)