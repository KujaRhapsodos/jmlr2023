from __future__ import annotations

import math
import numpy as np
import seaborn ; seaborn.set()
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
try:
    from sklearn.utils._testing import ignore_warnings
except ImportError:
    from sklearn.utils.testing import ignore_warnings
from tqdm import trange

from Model import Model
from Learners import Learner



class Pruner:
    """
    Prunes the RKHS weighting model learned by a Learner.

    Parameters
    ----------
    alpha : {'smart', 'safe', 'sure'} or float, default='smart'
        Parameter of the Lasso.
        'safe' uses a probabilistic bound to decide the value.
        'sure' uses an almost sure bound instead (always yields weaker pruning).
        'smart' will apply the Lasso multiple times to maximize pruning, with error difference < epsilon

    epsilon : float, default=0.01
        Desired maximal error difference between the original and pruned model.

    accept_criterion : {'ls', '01', 'sure bound', 'bound'}, default='01'
        Which stopping criterion to use when alpha='smart'.
        'ls' will stop when the training loss difference is larger than epsilon.
        '01' will stop when the training 01 error difference is larger than epsilon.
        'sure bound' will stop when the almost sure bound is larger than epsilon.
        'bound' will stop when the Lasso Rademacher bound is larger than epsilon.

    max_iter : int, default=10
        Maximal number of applications of the Lasso when alpha='smart'.

    n_cycles : int, default=5
        Determines the 'max_iter' parameter of the Lasso.
        Each variable will be seen up to 'n_cycles' times.

    verbose : int, default=1
        How much information to print when pruning. 
        0 will not print anything. 3 will print everything. 
        1 and 2 will print some.     
    """
    def __init__(self, alpha='smart', epsilon=0.01,
                 accept_criterion='01',
                 max_iter=10, n_cycles=5,
                 verbose=1):
        self.alpha = alpha
        self.epsilon = epsilon
        self.accept_criterion = accept_criterion
        self.max_iter = max_iter
        self.n_cycles = n_cycles
        self.verbose = verbose

    @ignore_warnings(category=ConvergenceWarning)
    def prune(self, learner: Learner):
        """
        Prunes the model by concentrating the coefficients around
        the most salient centers.

        Parameters
        ----------

        learner : Learner
            The model to be pruned is learner.model

        Note
        ----

        This function has no return. The learner.model is automatically
        changed to the newly pruned model.

        """
        self.learner = learner
        self.model_before = learner.model.copy()

        model_after, n_iter = self._smart_lasso_original(learner)

        if self.verbose > 1:
            print("Number of Lasso attempts : {}".format(n_iter))

        self._compare_before_and_after(learner, model_after)
        self.pruning_percent = self._get_pruning_percent(self.model_before, model_after)
        learner._set_model(model_after)

    def _get_candidate(self, model: Model, alpha: float) -> Model:
        raise NotImplementedError
    
    def _smart_lasso_original(self, learner: Learner):
        """
        Multiplies the lasso_param by 10 until rejection. Returns after that.
        """
        model = learner.model
        alpha = self._get_adjusted_alpha(learner)
        max_iter = self.max_iter if self.alpha == 'smart' else 1
        candidate = model.copy()
        model_after = candidate
        factor = 10
        n_iter = 0
        stop = False

        while n_iter < max_iter and model_after.get_n_centers() > 0 and not stop:
            if n_iter == 0:
                last_valid_alpha = alpha
                temp_alpha = last_valid_alpha
            else:
                temp_alpha = factor * last_valid_alpha
            candidate = self._get_candidate(model, temp_alpha)
            if self._is_candidate_acceptable(learner, candidate):
                model_after = candidate
                last_valid_alpha = temp_alpha
            else:
                stop = True
            n_iter += 1
        return model_after, n_iter

    def _get_adjusted_alpha(self, learner: Learner) -> float:
        raise NotImplementedError

    def _is_candidate_acceptable(self, learner: Learner, candidate: Model) -> bool:
        """
        Compares learner.model and candidate according to self.accept_criterion
        """
        criterion = self.accept_criterion.lower()
        if criterion == 'ls':
            loss_before = learner._model_training_loss(learner.model)
            loss_after = learner._model_training_loss(candidate)
            # print("Loss before, loss after : ", loss_before, loss_after)
            return loss_after - loss_before < self.epsilon
        elif criterion == '01':
            loss_before = learner._model_training_01_loss(learner.model)
            loss_after = learner._model_training_01_loss(candidate)
            return loss_after - loss_before < self.epsilon
        elif criterion == 'sure bound':
            bound = self.sure_bound(learner, candidate)
            return bound < self.epsilon
        elif criterion == 'bound':
            bound = self.rademacher_bound(learner, candidate)
            return bound < self.epsilon
        else:
            raise ValueError('Invalid accept_criterion')

    def _get_loss_difference(self, learner: Learner, candidate: Model):
        loss_before = learner._model_training_loss(learner.model)
        loss_after = learner._model_training_loss(candidate)
        return abs(loss_before - loss_after)

    def _get_alpha_increase_factor(self, learner: Learner, candidate: Model):
        loss_before = learner._model_training_loss(learner.model)
        loss_after = learner._model_training_loss(candidate)
        eps = self.epsilon
        val = eps / abs(loss_before - loss_after)
        return val

    def _compare_before_and_after(self, learner: Learner, model_after: Model):
        model = learner.model
        almost_sure_bound = self.sure_bound(model, model_after)
        rademacher_bound = self.rademacher_bound(model, model_after)
        if self.verbose >= 1:
            n_centers_before = model.get_n_centers()
            n_centers_after = model_after.get_n_centers()
            print('Lasso applied. Kept {} of {} terms.'.format(n_centers_after, n_centers_before))
        if self.verbose >= 3:
            loss_before = learner._model_training_loss(model)
            loss_after = learner._model_training_loss(model_after)
            loss_diff = loss_after - loss_before
            difference_rkhs_norm = (model - model_after).norm()
            print('RKHS norm of difference : {}'.format(difference_rkhs_norm))
            print('Training loss before : {}'.format(loss_before))
            print('Training loss after  : {}'.format(loss_after))
            print('Almost sure bound   : {}'.format(almost_sure_bound))
            print('Rademacher bound    : {}'.format(rademacher_bound))
            print('Epsilon             : {}'.format(self.epsilon))
            print('Actual loss difference  : {}'.format(loss_diff))

    def sure_bound(self, model_before: Model, model_after: Model) -> float:
        """
        Calculates and outputs the almost sure bound of the difference
        in risk before and after pruning:

        |L_D(before) - L_D(after)| < bound

        with 100% probability.
        """
        difference_rkhs_norm = (model_before - model_after).norm()
        return difference_rkhs_norm * self.learner._get_lipschitz() * model_before.operator_norm()

    def rademacher_bound(self, model_before: Model, model_after: Model) -> float:
        """
        Calculates and outputs a probabilistic bound of the difference
        in risk before and after pruning:

        |L_D(before) - L_D(after)| < bound

        with 1-delta probability.

        By default, delta=0.05.
        """
        learner = self.learner
        loss_before = learner._model_training_loss(model_before)
        loss_after = learner._model_training_loss(model_after)
        loss_diff = loss_after - loss_before
        difference_rkhs_norm = (model_before - model_after).norm()
        theta = learner.model.operator_norm()
        m = learner._get_training_size()
        delta = learner._get_delta()
        rho = learner._get_lipschitz()
        c = rho * theta * (2 + math.sqrt(2 * math.log(1 / delta)))
        return loss_diff + c * difference_rkhs_norm / math.sqrt(m)

    def _get_pruning_percent(self, model_before: Model, model_after: Model) -> float:
        n_centers_before = model_before.get_n_centers()
        n_centers_after = model_after.get_n_centers()
        return 100 * (n_centers_before - n_centers_after) / n_centers_before


class Method1Pruner(Pruner):
    """
    Prunes the RKHS weighting model learned by a Learner.

    Searches for a sparse weight function close in RKHS space to the original.

    Yields rather low pruning, but the resulting weight function will be similar to the original.

    Parameters
    ----------
    alpha : {'smart', 'safe', 'sure'} or float, default='smart'
        Parameter of the Lasso.
        'safe' uses a probabilistic bound to decide the value.
        'sure' uses an almost sure bound instead (always yields weaker pruning).
        'smart' will apply the Lasso multiple times to maximize pruning, with error difference < epsilon

    epsilon : float, default=0.01
        Desired maximal error difference between the original and pruned model.

    accept_criterion : {'ls', '01', 'sure bound', 'bound'}, default='01'
        Which stopping criterion to use when alpha='smart'.
        'ls' will stop when the training loss difference is larger than epsilon.
        '01' will stop when the training 01 error difference is larger than epsilon.
        'sure bound' will stop when the almost sure bound is larger than epsilon.
        'bound' will stop when the Lasso Rademacher bound is larger than epsilon.

    max_iter : int, default=10
        Maximal number of applications of the Lasso when alpha='smart'.

    n_cycles : int, default=5
        Determines the 'max_iter' parameter of the Lasso.
        Each variable will be seen up to 'n_cycles' times.

    verbose : int, default=1
        How much information to print when pruning. 
        0 will not print anything. 3 will print everything. 
        1 and 2 will print some.     
    """
    def __init__(self, alpha='smart', epsilon=0.01, 
                 accept_criterion='01', max_iter=10, n_cycles=5, verbose=1):
        super().__init__(alpha, epsilon, accept_criterion, max_iter, n_cycles, verbose)

    def _get_adjusted_alpha(self, learner: Learner) -> float:
        if type(self.alpha) in [int, float]:
            adjusted_alpha = float(self.alpha)
        if self.alpha == 'smart':
            adjusted_alpha = 1e-10
        if self.alpha == 'safe':
            adjusted_alpha = self._get_safe_alpha(learner)
        elif self.alpha == 'sure':
            adjusted_alpha = self._get_sure_alpha(learner)
        return adjusted_alpha

    def _get_sure_alpha(self, learner: Learner) -> float:
        n_centers = learner.model.get_n_centers()
        rho = learner._get_lipschitz()
        theta = learner.model.operator_norm()
        eps = self.epsilon
        coef_l1_norm = np.sum(np.abs(learner.model.coefs))
        adjusted_param = eps**2 / (12 * n_centers * rho**2 * theta**2 * coef_l1_norm)
        if self.verbose >= 2:
            print('sure lasso param : {}'.format(adjusted_param))
        return adjusted_param

    def _get_safe_alpha(self, learner: Learner) -> float:
        n_centers = learner.model.get_n_centers()
        m = learner._get_training_size()
        rho = learner._get_lipschitz()
        theta = learner.model.operator_norm()
        delta = learner._get_delta()
        coef_l1_norm = np.sum(np.abs(learner.model.coefs))
        c = rho * theta * (2 + math.sqrt(2 * math.log(1 / delta)))
        adjusted_param = m * self.epsilon**2 / (12 * n_centers * coef_l1_norm * c**2)
        if self.verbose >= 2:
            print('safe lasso param : {}'.format(adjusted_param))
        return adjusted_param

    def _get_candidate(self, model: Model, alpha: float) -> Model:
        """
        Returns a new Model object obtained from applying the Lasso procedure.

        The Lasso objective is the RKHS norm plus an l1 regularizer.
        """
        T = model.get_n_centers()
        if T == 0:
            return model
        else:
            coef_before = np.array(model.coefs)
            cho_up = model.choleski_upper()     
            lasso = Lasso(alpha=alpha, fit_intercept=False, warm_start=True, 
                        tol=1e-4, max_iter=self.n_cycles * T)
            lasso.coef_ = coef_before.copy() # warm start
            lasso.fit(cho_up, np.dot(cho_up, coef_before.T))
            coef_after = lasso.coef_.copy().tolist()
            return model.copy().set_coefs(coef_after)
