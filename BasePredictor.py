import numpy as np

class BasePredictor:
    def eval(self, W, X) -> np.ndarray:
        """
        Return the n_examples x n_parameters array of the
        base predictor on every pair (example, parameter).
        """
        raise NotImplementedError

class Sign(BasePredictor):
    def eval(self, W, X) -> np.ndarray:
        W = np.array(W)
        output = np.sign(np.dot(X, W.T))
        output[output == 0] = 1
        return output

class Relu(BasePredictor):
    def eval(self, W, X) -> np.ndarray:
        W = np.array(W)
        output = np.dot(X, W.T)
        output[output < 0] = 0
        return output

class Stump(BasePredictor):
    def eval(self, W, X) -> np.ndarray:
        W = np.array(W).reshape(-1, 2)
        idx = W[:, 0].astype(int)
        value = W[:, 1]
        output = np.sign(X[:, idx] - value)
        output[output == 0] = 1
        return output