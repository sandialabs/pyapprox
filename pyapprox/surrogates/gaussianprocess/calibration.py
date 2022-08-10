import numpy as np
import copy
from scipy.optimize import approx_fprime
from scipy.linalg import cho_solve
from functools import partial

from pyapprox.surrogates.gaussianprocess.multilevel_gp import MultilevelGP
from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.bayes.markov_chain_monte_carlo import MCMCVariable


class CalibrationGP(MultilevelGP):
    def set_data(self, samples, values, theta):
        # when length_scale bounds is fixed then the V2(D2) block in Kennedys
        # paper should always be the same regardless of value of theta
        # Currently I have extra length scales for V2 kernel, i.e. for
        # theta dimensions. This will not effect answer given other
        # increase size of optimization problem
        samples_copy = copy.deepcopy(samples)
        samples_copy[-1] = get_all_sample_combinations(samples[-1], theta)
        super().set_data(samples_copy, values)

    def plot_1d(self, theta, num_XX_test, bounds,
                ax=None, num_stdev=2, plt_kwargs={}, fill_kwargs={},
                prior_fill_kwargs=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[None, :]
        # return_std=True does not work for gradient enhanced krigging
        # gp_mean, gp_std = predict(XX_test,return_std=True)
        XX = get_all_sample_combinations(XX_test, theta)
        gp_mean, gp_std = self(XX, return_std=True)
        gp_mean = gp_mean[:, 0]
        if prior_fill_kwargs is not None:
            prior_std = np.sqrt(self.kernel_.diag(XX_test.T))
            ax.fill_between(
                XX_test[0, :], self._y_train_mean-num_stdev*prior_std,
                self._y_train_mean+num_stdev*prior_std, **prior_fill_kwargs)
        ax.plot(XX_test[0, :], gp_mean, **plt_kwargs)
        ax.fill_between(
            XX_test[0, :], gp_mean-num_stdev*gp_std, gp_mean+num_stdev*gp_std,
            **fill_kwargs)
        return ax


def _gp_negloglike(kernel, train_samples, train_values, theta, **gp_kwargs):
    gp = CalibrationGP(kernel, **gp_kwargs)
    assert theta.ndim == 2 and theta.shape[1] == 1
    gp.set_data(
        train_samples, train_values, theta)
    gp.fit()
    # print(gp.kernel_)
    data = np.vstack(train_values)
    val = data.T.dot(
        cho_solve((gp.L_, True), data, check_finite=False))
    # det LL.T is diag(L)**2 so
    # 1/2 log diag(LL.T)=2/2*sum(log(diag(L)))
    val += np.log(np.diag(gp.L_)).sum()
    # d = gp.L_.shape[0]
    # val += d/2*np.log(2*np.pi)
    return val


def _gp_loglike(negloglike, t, jac=False):
    val = -negloglike(t)
    grad = -approx_fprime(
        t[:, 0], lambda s: negloglike(s[:, None])[:, 0],
        10*np.sqrt(np.finfo(float).eps))
    if jac:
        return val, grad
    return val


class GPCalibrationVariable(MCMCVariable):
    # TODO create new MCMCVariable base class without normalized pdf etc
    # and inherit this from that here and for old MCMCVariable
    def __init__(self, variable, kernel, train_samples, train_values,
                 algorithm="metropolis", **gp_kwargs):
        # estimate hypeprameters using variable.mean()
        self.gp = CalibrationGP(kernel, **gp_kwargs)
        self.gp.set_data(train_samples, train_values,
                         variable.get_statistics("mean"))
        self.gp.fit()
        print(self.gp.kernel_)
        self.gp.kernel_.length_scale_bounds = "fixed"

        negloglike = partial(
            _gp_negloglike, self.gp.kernel_, train_samples, train_values,
            **gp_kwargs)
        loglike = partial(_gp_loglike, negloglike, jac=False)
        super().__init__(variable, loglike, algorithm, loglike_grad=True,
                         burn_fraction=0.1, njobs=1)

    def __str__(self):
        string = "GPCalibrationVariable with prior:\n"
        string += self._variable.__str__()
        return string
