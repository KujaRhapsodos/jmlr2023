import numpy as np
from scipy.spatial.distance import cdist

class Kernel:
    def __init__(self) -> None:
        pass

    def calculate(self, W, U) -> np.ndarray:
        raise NotImplementedError
    
    def diag(self, W, U) -> np.ndarray:
        """
        Calculates and returns only the diagonal of the kernel matrix, i.e.
        np.diag(self.calculate(W, U)).

        W and U must be of the same dimensions.
        """
        diag_values = np.zeros(len(W))
        for i, (w, u) in enumerate(zip(W, U)):
            diag_values[i] = self.calculate([w], [u])
        return diag_values
    

class GaussianKernel(Kernel):
    def __init__(self, gamma=1.0) -> None:
        self.gamma = gamma

    def calculate(self, W, U) -> np.ndarray:
        W = np.array(W)
        W = W.reshape(W.shape[0], -1)
        U = np.array(U)
        U = U.reshape(U.shape[0], -1)
        dist = cdist(W, U, 'sqeuclidean')
        return np.exp(-dist / (2 * self.gamma**2))

class IndicatorGaussianKernel(Kernel):
    def __init__(self, gamma=1.0) -> None:
        self.gamma = gamma

    def calculate(self, W, U) -> np.ndarray:
        W_idx = np.array([idx for (idx, val) in W])
        W_val = np.array([val for (idx, val) in W]).reshape(-1, 1)
        U_idx = np.array([idx for (idx, val) in U])
        U_val = np.array([val for (idx, val) in U]).reshape(-1, 1)
        indicator = (W_idx[:, None] == U_idx) * 1
        dist = cdist(W_val, U_val, 'sqeuclidean')
        return np.multiply(indicator, np.exp(-dist / (2 * self.gamma**2)))
