import numpy as np

class Loss:
    def __init__(self) -> None:
        pass

    def calculate(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def lipschitz(self, max_model_output: float) -> float:
        raise NotImplementedError

    def max_value(self, max_model_output: float) -> float:
        raise NotImplementedError

    def min_value(self, max_model_output: float) -> float:
        raise NotImplementedError

class LogisticLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        self.large = 50

    def calculate(self, output: np.ndarray, targets: np.ndarray) -> float:
        margins = -np.multiply(targets, output)
        not_too_large_idx = margins < self.large
        loss = margins
        loss[not_too_large_idx] = np.log(1 + np.exp(loss[not_too_large_idx]))
        return np.mean(loss)

    def derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return self.stable_derivative(output, targets)

    def stable_derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert(output.shape == targets.shape)
        margins = -np.multiply(targets, output)
        adjustment = margins.copy()
        adjustment[adjustment < 0] = 0
        exp = np.exp(margins - adjustment)
        df = -targets * exp / (np.exp(-adjustment) + exp)
        return df

    def unstable_derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        assert(output.shape == targets.shape)
        margins = -np.multiply(targets, output)
        not_too_large_idx = margins < self.large
        df = np.ones(margins.shape[0])
        exp = np.exp(margins[not_too_large_idx])
        df[not_too_large_idx] = -targets[not_too_large_idx] * exp / (1 + exp)
        return df

    def lipschitz(self, max_model_output: float) -> float:
        return 1.0

    def max_value(self, max_model_output: float) -> float:
        return max_model_output if max_model_output > 50 else np.log(1 + np.exp(max_model_output))

    def min_value(self, max_model_output: float) -> float:
        return 0.0


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, output: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((output - targets)**2)

    def derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return 2*(output - targets)

    def max_value(self, max_model_output: float) -> float:
        return (max_model_output + 1.0)**2

    def min_value(self, max_model_output: float) -> float:
        return 0.0

    def lipschitz(self, max_model_output: float) -> float:
        return 2*(max_model_output + 1)

class Hinge(Loss):
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, output: np.ndarray, targets: np.ndarray) -> float:
        loss = 1 - np.multiply(targets, output)
        loss[loss < 0] = 0
        return np.mean(loss)

    def derivative(self, output: np.ndarray, targets: np.ndarray) -> np.ndarray:
        margins = np.multiply(targets, output)
        df = np.zeros(margins.shape[0])
        df[margins < 1] = -targets[margins < 1]
        return df

    def lipschitz(self, max_model_output: float) -> float:
        return 1.0

    def max_value(self, max_model_output: float) -> float:
        return max_model_output + 1

    def min_value(self, max_model_output: float) -> float:
        return 0.0
