from __future__ import annotations

import math
import numpy as np
import scipy as sp
import seaborn ; seaborn.set()
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import time
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from tqdm import trange

from Model import Model
from Loss import *

class Learner(BaseEstimator, ClassifierMixin):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    model : Model
        Model to be learned.

    loss : Loss
        Optimisation loss.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    Attributes
    ----------
    model : Model
        The learned prediction model.
    """
    def __init__(self, model: Model, loss: Loss, rng=None):
        self.model = model
        self.loss = loss
        self.rng = rng

    def _more_tags(self):
        return {'binary_only': True}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Learner: 
        raise NotImplementedError

    def _preprocessing(self, X: np.ndarray, y: np.ndarray):
        self.rng = np.random.default_rng(self.rng)
        X, y = check_X_y(X, y)
        self.training_data = X
        self.classes_, self.training_targets = self._classes_preprocessing(y)

    def _classes_preprocessing(self, y: np.ndarray):
        self.y_ = y
        self.classes_ = np.unique(y)
        assert(len(self.classes_) == 2)
        self.training_targets = self._classes_to_targets(y)
        return self.classes_, self.training_targets

    def _classes_to_targets(self, classes: np.ndarray) -> np.ndarray:
        targets = classes.copy()
        targets[classes == self.classes_[0]] = 1
        targets[classes == self.classes_[1]] = -1
        return targets

    def _targets_to_classes(self, targets: np.ndarray) -> np.ndarray:
        classes = targets.copy()
        classes[targets == 1] = self.classes_[0]
        classes[targets == -1] = self.classes_[1]
        return classes

    def _get_n_features(self) -> int:
        return self.training_data.shape[1]
    
    def _get_lipschitz(self) -> float:
        return self.loss.lipschitz(self.model.max_output())

    def _get_training_size(self) -> int:
        return self.training_data.shape[0]

    def _get_delta(self) -> float:
        return 0.05

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        proba = self.predict_proba(X)
        return self._proba_to_classes(proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba_from_output(self.output(X))

    def _proba_from_output(self, output: np.ndarray) -> np.ndarray:
        return self._logistic_function(output)

    def _logistic_function(self, array: np.ndarray) -> np.ndarray:
        return sp.special.expit(array) 

    def _proba_to_classes(self, proba: np.ndarray) -> np.ndarray:
        classes = np.zeros(proba.shape, self.classes_.dtype)
        classes[proba > 0.5] = self.classes_[0]
        classes[proba <= 0.5] = self.classes_[1]
        return classes

    def output(self, X) -> np.ndarray:
        """
        Returns the size m array containing
        the output of the prediction model for all examples in X.

        Equations-wise, this is Lambda alpha(X).
        """
        return self.model.output(X)    

    def _sample_batch(self, batch_size):
        sample_size, n_features = self.training_data.shape
        batch_idx = self.rng.choice(sample_size, size=batch_size, replace=True)
        batch_data = self.training_data[batch_idx, :].reshape((batch_size, n_features))
        batch_targets = self.training_targets[batch_idx]
        return batch_data, batch_targets 

    def rademacher_bound(self) -> float:
        """
        Probabilistic bound on the generalization error.
        """
        norm = self.model.norm()
        max_output = self.model.max_output()
        m = self._get_training_size()
        rho = self._get_lipschitz()
        theta = self.model.operator_norm()
        delta = self._get_delta()
        max_loss = self.loss.max_value(max_output)
        min_loss = self.loss.min_value(max_output)
        bound = self.training_loss()
        bound += 2 * norm * rho * theta / math.sqrt(m)
        bound += (max_loss-min_loss) * math.sqrt(math.log(1 / delta) / (2 * m))
        return bound

    def _probabilistic_rademacher_bound_without_mcdiarmid(self) -> float:
        """
        Probabilistic bound on the generalization error.
        Independent of the maximal value of the loss.     
        Loose and mostly deprecated.
        """
        return self.rademacher_bound_without_mcdiarmid() / self._get_delta()

    def rademacher_bound_without_mcdiarmid(self) -> float:
        """
        Expectation bound on the generalization error.
        Independent of the maximal value of the loss.     
        Loose and mostly deprecated.
        """
        norm = self.model.norm()
        m = self._get_training_size()
        rho = self._get_lipschitz()
        theta = self.model.operator_norm()
        bound = 2 * norm * rho * theta
        bound /= math.sqrt(m)
        bound += self.training_loss()
        return bound

    def _model_training_loss(self, model: Model) -> np.ndarray:
        output = model.output(self.training_data)
        targets = self.training_targets
        return self.loss.calculate(output, targets)

    def _model_training_01_loss(self, model: Model) -> np.ndarray:
        proba = self._proba_from_output(model.output(self.training_data))
        pred = self._proba_to_classes(proba)
        return 1 - accuracy_score(self.y_, pred)

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        output = self.output(X)
        targets = self._classes_to_targets(y)
        return self.loss.calculate(output, targets)

    def training_loss(self) -> np.ndarray:
        return self._model_training_loss(self.model)

    def training_01_loss(self) -> np.ndarray:
        return self._model_training_01_loss(self.model)

    def _get_max_training_norm(self) -> float:
        try: 
            return self.max_training_norm
        except:
            norms = np.linalg.norm(self.training_data, axis=1)
            self.max_training_norm = max(norms)
            return self.max_training_norm

    def _set_model(self, new_model: Model):
        """
        Utility function that directly sets self.model to new_model.
        """
        self.model = new_model
    

class SFGDLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    model : Model
        Model to be learned.

    loss : Loss
        Optimisation loss.

    n_iter : int, default=100
        Number of iterations of the stochastic functional gradient descent

    B : {'auto'} or int, default=1000
        Maximal RKHS norm of the weight function.
        If 'auto', then B = sqrt(m)/theta.

    regularization : {'auto'} or float, default='auto'
        Tikhonov regularization parameter.
        If 'auto', then it will be equal to the value suggested by the convergence bounds.

    stepsize : {'auto'} or float, default='auto'
        Stepsize of the gradient descent.
        If 'auto', then it will be equal to the value suggested by the convergence bounds.

    batch_size : int, default=32
        Number of examples sampled every iteration for approximating the functional gradient.

    apply_projection_step : boolean, default=True
        Whether to bound by B the iterates of the gradient descent.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    Attributes
    ----------
    model : Model
        The learned prediction model.
    
    """
    def __init__(self, model: Model, loss=LogisticLoss(), 
                 n_iter=100, B=1000, regularization='auto', 
                 stepsize='auto', batch_size=32,
                 apply_projection_step=True, 
                 rng=None, **kwargs):
        super().__init__(model=model, loss=loss, rng=rng)
        self.n_iter = n_iter
        self.B = B
        self.regularization = regularization
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.apply_projection_step = apply_projection_step

    def fit(self, X: np.ndarray, y: np.ndarray) -> SFGDLearner: 
        self._preprocessing(X, y)
        self._initialize_values()
        self.iterate(self.n_iter)
        return self

    def _initialize_values(self):
        self.iteration_model = self.model.copy()
        self.n_total_iter_ = 0
        self.n_of_average_ = 1
        self.iterate_norms = [0.0]

    def _set_loss(self, loss: str):
        """
        Utility function for calculating different losses.

        Use carefully.
        """
        if loss.lower() == 'logistic':
            self.loss = LogisticLoss()
        elif loss.lower() == 'mse':
            self.loss = MSE()
        elif loss.lower() == 'hinge':
            self.loss = Hinge()

    def iterate(self, n_iter):
        for _ in range(n_iter):
            self._take_one_step(self.batch_size)
    
    def _take_one_step(self, batch_size):
        center = self.iteration_model.sample_center()
        batch_data, batch_targets = self._sample_batch(batch_size)
        batch_output = self.iteration_model.output(batch_data)
        derivative = self.loss.derivative(batch_output, batch_targets)
        phi_of_x = self.iteration_model.base_pred.eval(center, batch_data)
        coef = -self._get_stepsize() * derivative * phi_of_x.flatten()
        scale = 1.0 - self._get_stepsize() * self._get_regularization()
        self.iteration_model.efficient_update(center, np.mean(coef), scale)
        if self.apply_projection_step:
            self.iteration_model.project(self._get_B())
        self.model.update_average(self.iteration_model, self.n_of_average_)
        self.iterate_norms.append(self.iteration_model.norm())
        self.n_total_iter_ += 1
        self.n_of_average_ += 1 

    # TODO Shouldn't I make it so all constants are calculated once and then fetched from memory?
    def _get_B(self) -> float:
        if self.B == 'auto':
            return math.sqrt(self._get_training_size()) / self.model.operator_norm()
        else:
            return float(self.B)
    
    def _get_stepsize(self) -> float:
        if self._use_slow_sgd() and self.stepsize == 'auto':
            return self._get_B() / (self._get_lipschitz() * self.model.kappa() * math.sqrt(self.n_iter))
        elif self.stepsize == 'auto':
            reg = self._get_regularization()
            return 1 / (reg * (self.n_total_iter_ + 1))
        else:
            return float(self.stepsize)
    
    def _get_regularization(self) -> float:
        if self._use_slow_sgd():
            return 0
        elif self.regularization == 'auto':
            rho = self._get_lipschitz()
            B = self._get_B()
            m = self._get_training_size()
            return math.sqrt(8) * rho / (B * math.sqrt(m))
        else:
            return float(self.regularization)

    def _get_lipschitz(self) -> float:
        # Importantly, it must be self.iteration_model below rather than self.model.
        # Otherwise, a huge slowdown will happen because the norm of self.model is
        # not efficiently updated every iteration.
        return self.loss.lipschitz(self.iteration_model.max_output())

    def _get_max_iterate_norm(self) -> float:
        return max(self.iterate_norms)

    def _use_slow_sgd(self) -> bool:
        return self.regularization == 0

    def probabilistic_stability_bound(self):
        """
        Probabilistic bound on the convergence of the gradient descent.
        """
        if not self._is_stability_bound_valid():
            return "Stability bound is invalid."
        else:
            return self.stability_bound() / self._get_delta()

    def stability_bound(self):
        """
        Bound in expectation on the convergence of the gradient descent.
        """
        reg = self._get_regularization()
        if not self._is_stability_bound_valid():
            return "Stability bound is invalid."
        B = self._get_max_iterate_norm()
        m = self._get_training_size()
        T = self.n_total_iter_
        rho = self._get_lipschitz()
        theta = self.model.operator_norm()
        kappa = self.model.kappa()
        term1 = 2 * rho * theta * B / math.sqrt(m)
        term2 = reg * B**2
        term3 = 8 * rho**2 / (reg * m)
        term4 = (1 + math.log(T)) * (rho * kappa + reg * B)**2 / (2 * reg * T)
        bound = term1 + term2 + term3 + term4
        return bound

    def _is_stability_bound_valid(self):
        return self._get_regularization() > 0 and self.stepsize == 'auto'

    def probabilistic_slow_sgd_bound(self):
        if self._use_slow_sgd() and self.stepsize == 'auto':
            return self._get_B() * self._get_lipschitz() * self.model.kappa() / math.sqrt(self.n_iter) / self._get_delta()
        else:
            return 'Slow SGD bound does not apply.'

    def _set_model(self, new_model: Model):
        """
        Utility function that directly sets self.model to new_model.
        """
        self.model = new_model
        self.iteration_model = new_model
        self.n_total_iter_ = 0
        self.n_of_average_ = 1
        self.iterate_norms.append(new_model.norm())


class LeastSquaresLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    model : Model
        Model to be learned.

    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    Attributes
    ----------
    model : Model
        The learned prediction model.
    
    """
    def __init__(self, model: Model, n_iter=100, regularization=1e-5, rng=None, **kwargs):
        super().__init__(model=model, loss=MSE(), rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization

    def fit(self, X: np.ndarray, y: np.ndarray, centers=None) -> LeastSquaresLearner: 
        self._preprocessing(X, y)
        self._least_squares_fit(centers=centers)
        return self
    
    def _least_squares_fit(self, centers=None):
        """
        Learn the optimal coefficients.

        The regularization term is the RKHS norm of the weight function.
        """
        if centers is None:
            centers = [self.model.sample_center() for _ in range(self.n_iter)]

        self.model.set_centers(centers)
        Phi = self.model.expectations(self.training_data)
        G = self.model.gram()
        m, _ = self.training_data.shape 
        
        A = np.dot(Phi.T, Phi) + m * self.regularization * G
        B = np.dot(Phi.T, self.training_targets)

        coefs = np.linalg.solve(A, B)

        self.model.set_coefs(coefs.tolist())

    def _rks_fit(self, centers=None):
        """
        Learn the optimal coefficients.

        The regularization term is the euclidean norm of the weight function coefficients.
        """
        if centers is None:
            centers = [self.model.sample_center() for _ in range(self.n_iter)]
            
        self.model.set_centers(centers)

        Phi = self.model.expectations(self.training_data)
        m, _ = self.training_data.shape 
        T = len(centers)
        
        A = np.dot(Phi.T, Phi) + m * self.regularization() * np.eye(T)
        B = np.dot(Phi.T, self.training_targets)

        coefs = np.linalg.solve(A, B)

        self.model.set_coefs(coefs.tolist())


class LassoLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    model : Model
        Model to be learned.

    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    Attributes
    ----------
    model : Model
        The learned prediction model.
    
    """
    def __init__(self, model: Model, n_iter=100, regularization=1e-5, rng=None, **kwargs):
        super().__init__(model=model, loss=MSE(), rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization

    def fit(self, X: np.ndarray, y: np.ndarray) -> LeastSquaresLearner: 
        self._preprocessing(X, y)
        self._lasso_fit()
        return self
        
    @ignore_warnings(category=ConvergenceWarning)
    def _lasso_fit(self, centers=None):
        if centers is None:
            centers = [self.model.sample_center() for _ in range(self.n_iter)]
            
        self.model.set_centers(centers)

        Phi = self.model.expectations(self.training_data)
        
        lasso = Lasso(alpha=self.regularization, fit_intercept=False, max_iter=5*len(centers))
        lasso.fit(Phi, self.training_targets)
        coefs = lasso.coef_.copy().tolist()

        self.model.set_coefs(coefs)
        self.model.remove_useless_centers()

# TODO Very slow, figure out why.
class OptimalStepsizeLearner(Learner):
    """
    Classify using RKHS weightings of functions.

    Parameters
    ----------
    model : Model
        Model to be learned.

    n_iter : int, default=100
        Number of sampled centers.

    regularization : float, default=1e-5
        Tikhonov regularization parameter.
    
    rng : Numpy Random Generator or int or None
        Random number generator or random seed to set the randomness.

    Attributes
    ----------
    model : Model
        The learned prediction model.
    
    """
    def __init__(self, model: Model, loss=MSE(), n_iter=100, regularization=1e-5, 
                 use_batch=True, batch_size=100, rng=None, **kwargs):
        super().__init__(model=model, loss=loss, rng=rng)
        self.n_iter = n_iter
        self.regularization = regularization
        self.use_batch = use_batch
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> OptimalStepsizeLearner: 
        self._preprocessing(X, y)
        self.iterate(self.n_iter)
        return self

    def iterate(self, n_iter):
        for _ in range(n_iter):
            self._take_one_step()
    
    def _take_one_step(self):
        center = self.model.sample_center()
        coef = self._get_coef(center)
        self.model = (1-self.regularization) * self.model
        self.model.add_center(center, coef)
    
    def _get_coef(self, center) -> float:
        # TODO Attempt to optimize. Center_model definition is surprisingly costly.
        center_model = self.model.copy().set_centers([center], [1])
        if self.use_batch:
            batch_data, batch_targets = self._sample_batch(self.batch_size)
        else:
            batch_data, batch_targets = self.training_data, self.training_targets

        model_outputs = self.model.output(batch_data)
        center_model_outputs = center_model.output(batch_data)
        alpha_of_w = self.model.eval_weight_func(center)
        reg = self.regularization
        Kww = center_model.norm()**2
        m, _ = batch_data.shape

        if isinstance(self.loss, MSE):
            num = 1/m * np.dot((1-reg) * model_outputs - batch_targets, center_model_outputs) + reg * (1-reg) * alpha_of_w
            denom = 1/m * np.sum(center_model_outputs**2) + reg * Kww
            return -num / denom
        else:
            loss = self.loss

            def derivative_function(eta):
                value = reg * (1-reg) * (alpha_of_w + eta * Kww)
                partial_derivatives = loss.derivative((1-reg)*model_outputs + eta * center_model_outputs, batch_targets)
                value += 1/m * np.dot(partial_derivatives, center_model_outputs)
                return value

            return sp.optimize.root_scalar(derivative_function, x0=0, x1=50, maxiter=None).root

class CVLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, learner_class: Learner, model_class: Model, 
                 learner_param_grid: dict, model_param_grid: dict, folds=5, rng=None) -> None:
        self.learner_class = learner_class
        self.model_class = model_class
        self.learner_param_grid = learner_param_grid
        self.model_param_grid = model_param_grid
        self.folds = folds
        self.rng = rng

    def fit(self, X: np.ndarray, y: np.ndarray) -> CVLearner:
        self.best_score_ = 0
        self.best_estimator_ = None
        self.best_learner_params_ = None
        self.best_model_params_ = None
        for learner_params in ParameterGrid(self.learner_param_grid):
            for model_params in ParameterGrid(self.model_param_grid):  
                model = self.model_class(input=X, **model_params, rng=self.rng)
                clf = self.learner_class(model=model, **learner_params, rng=self.rng)
                score = self.avg_cv_score(clf, X, y)
                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_estimator_ = clf
                    self.best_learner_params_ = learner_params
                    self.best_model_params_ = model_params
        model = self.model_class(input=X, **self.best_model_params_)
        clf = self.learner_class(model=model, **self.best_learner_params_)
        start_refit = time.time()
        clf.fit(X, y)
        self.refit_time_ = time.time() - start_refit
        self.best_estimator_ = clf
        return self
        
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def avg_cv_score(self, clf: Learner, X: np.ndarray, y: np.ndarray) -> float:
        kf = KFold(n_splits=self.folds, shuffle=True, random_state=0) 
        total_score = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(X_train, y_train)
            total_score += clf.score(X_test, y_test)
        return total_score / self.folds
        