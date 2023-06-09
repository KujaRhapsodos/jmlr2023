from __future__ import annotations

from copy import deepcopy
from typing import List
import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
import math

from Kernel import *

class WeightFunction:

    def __init__(self, kernel: Kernel=GaussianKernel()):
        self.kernel = kernel
        self.list_of_centers = []
        self.coefs = []
        self.running_norm = 0
        self.norm_is_up_to_date = True

    def add_center(self, center, coef) -> WeightFunction:
        if coef != 0:
            if len(self.list_of_centers) == 0:
                self.list_of_centers.append(center)
                self.coefs.append(coef)
                self.running_norm = self.calculate_rkhs_norm()
            else:
                self += WeightFunction(self.kernel).add_center(center, coef)
        return self
    
    def calculate_rkhs_norm(self) -> float:
        """
        Updates and returns the RKHS norm of the weight function.
        """
        self.running_norm = math.sqrt(self.scalar_product(self))
        self.norm_is_up_to_date = True
        return self.running_norm

    def scalar_product(self, beta: WeightFunction) -> np.ndarray:
        if self.get_n_centers() == 0 or beta.get_n_centers() == 0:
            return 0.0
        gram = self.kernel.calculate(self.list_of_centers, beta.list_of_centers)
        return np.dot(self.coefs, np.dot(gram, beta.coefs))

    def get_n_centers(self) -> int:
        return len(self.list_of_centers)

    def __add__(self, beta: WeightFunction) -> WeightFunction:
        new = self.copy()
        new.efficient_norm_update(beta)
        new.list_of_centers = self.list_of_centers + beta.list_of_centers
        new.coefs = self.coefs + beta.coefs
        return new

    def __sub__(self, beta: WeightFunction) -> WeightFunction:
        return self + (-beta)

    def __neg__(self) -> WeightFunction:
        new = self.copy()
        new *= -1
        return new

    def __iadd__(self, beta: WeightFunction) -> WeightFunction:
        self.efficient_norm_update(beta)
        self.list_of_centers = self.list_of_centers + beta.list_of_centers
        self.coefs = self.coefs + beta.coefs
        return self

    def __rmul__(self, factor) -> WeightFunction:   
        """
        Multiply all coefs by factor, and updates the running norm.
        """
        new = self.copy()
        new.coefs = (factor * np.array(new.coefs)).tolist()
        new.running_norm = factor * new.running_norm
        return new

    def __imul__(self, factor) -> WeightFunction:
        """
        Multiply all coefs by factor, and updates the running norm.
        """
        self.coefs = (factor * np.array(self.coefs)).tolist()
        self.running_norm = factor * self.running_norm
        return self

    def __truediv__(self, factor) -> WeightFunction:
        """
        Divide all coefs by factor, and updates the running norm.

        Simply returns self if factor is 0.
        """
        if factor != 0:
            self.coefs = (np.array(self.coefs) / factor).tolist()
            self.running_norm = self.running_norm / factor
        return self


    def efficient_norm_update(self, beta: WeightFunction) -> None:
        """
        Updates the running norm given the weight function
        that has newly been added (i.e. the norm of self + beta).
        Much more efficient than self.calculate_rkhs_norm if beta
        has few centers.
        """
        scalar = self.scalar_product(beta)
        norm_squared = self.norm()**2 + 2 * scalar + beta.norm()**2
        try:
            self.running_norm = math.sqrt(norm_squared)
        except ValueError:
            # Due to imprecision, value will sometimes be slightly negative instead of 0.
            self.running_norm = math.sqrt(abs(round(norm_squared, 10)))
        self.norm_is_up_to_date = True
    
    def norm(self) -> float:
        """
        Efficiently returns the RKHS norm of the weight function.
        Will compute only if necessary.
        """
        return self.running_norm if self.norm_is_up_to_date else self.calculate_rkhs_norm()

    def efficient_add(self, beta: WeightFunction, factor=1.0) -> WeightFunction:
        """
        Efficient sum when only the last center is different.
        Yields (self + factor * beta).
        Used for calculating the running average of the model.
        Running RKHS norm is not updated.
        """
        for i in range(len(beta.list_of_centers)):
            new_coef = factor * beta.coefs[i]
            if i < len(self.list_of_centers):
                self.coefs[i] += new_coef
            else:
                self.list_of_centers.append(beta.list_of_centers[i])
                self.coefs.append(new_coef)
        self.norm_is_up_to_date = False
        return self

    def update_average(self, beta: WeightFunction, n_iter) -> WeightFunction:
        """
        Assumes self is a running average of n_iter models, and updates
        using the formula :

        self = n_iter / (1 + n_iter) * self + 1 / (1 + n_iter) * beta

        Assumes beta has a single additionnal center.
        """
        self *= n_iter / (1 + n_iter)
        factor = 1 / (1 + n_iter)
        for i in range(len(beta.list_of_centers)):
            new_coef = factor * beta.coefs[i]
            if i < len(self.list_of_centers):
                self.coefs[i] += new_coef
            else:
                self.list_of_centers.append(beta.list_of_centers[i])
                self.coefs.append(new_coef)
        self.norm_is_up_to_date = False
        return self

    def copy(self) -> WeightFunction:
        return deepcopy(self)

    def set_centers(self, centers: List, coefs=None) -> WeightFunction:
        """
        Overwrites the list of centers.
        Used for arbitrarily changing the weight function.
        """
        self.list_of_centers = centers
        if coefs is None:
            self.coefs = [0] * len(centers)
        else:
            self.coefs = coefs
            self.remove_useless_centers()
        self.norm_is_up_to_date = False
        return self

    def set_coefs(self, coefs: List) -> WeightFunction:
        """
        Overwrites the coefficients without changing the centers.
        Used after applying the Lasso.
        """
        self.coefs = coefs
        self.remove_useless_centers()
        self.norm_is_up_to_date = False
        return self

    def project(self, max_norm) -> float:
        """
        Ensures the RKHS norm is at most max_norm.
        """
        factor = min(1.0, max_norm / self.norm()) if self.norm() else 1.0 
        self *= factor
        return factor

    def remove_useless_centers(self) -> WeightFunction:
        """
        Removes centers which have a zero coefficient.
        """
        n_centers = self.get_n_centers()
        nonzero_idx = np.flatnonzero(self.coefs)
        if len(nonzero_idx) < n_centers:
            self.coefs = np.array(self.coefs)[nonzero_idx].tolist()
            self.list_of_centers = np.array(self.list_of_centers, dtype=object)[nonzero_idx, :].tolist()
        return self

    def gram(self) -> np.ndarray:
        """
        Returns the Gram matrix of the centers.
        """
        return self.kernel.calculate(self.list_of_centers, self.list_of_centers)

    def get_max_center_norm(self) -> float:
        """
        Returns the square root of the maximal value of K(w, w)
        for all w's in the list of centers.
        """
        return math.sqrt(np.max(np.diag(self.gram())))

    def choleski_upper(self) -> np.ndarray:
        n_centers = self.get_n_centers()
        gram = self.gram()
        try:
            choleski_upper = cholesky(gram)
        except:
            choleski_upper = cholesky(gram + 1e-8 * np.identity(n_centers)) 
        return choleski_upper  

    def merge_duplicate_centers(self) -> WeightFunction:
        """
        Merges the coefficients for centers that are equal.
        This is a somewhat costly operation.
        Do not use every iteration.
        """
        n = self.get_n_centers()
        if n > 0:
            dist = cdist(self.list_of_centers, self.list_of_centers, 'sqeuclidean')
            remaining_idx = list(range(n))
            unique_idx = []
            unique_coefs = []
            while len(remaining_idx) > 0:
                idx = remaining_idx[0]
                coef = 0
                for j in range(idx, n):
                    if dist[idx, j] == 0:
                        coef += self.coefs[j]
                        remaining_idx.remove(j)
                unique_idx.append(idx)
                unique_coefs.append(coef)   
            self.list_of_centers = np.array(self.list_of_centers)[unique_idx, :].tolist()
            self.coefs = unique_coefs             
        return self
   
    def eval_weight_func(self, w) -> float:
        beta = WeightFunction(self.kernel).add_center(w, 1)
        return self.scalar_product(beta)

    def eval_weight_func_multiple_centers(self, list_of_w: list) -> np.ndarray:
        return [self.eval_weight_func(w) for w in list_of_w]
