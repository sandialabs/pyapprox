import numpy as np
import copy
from scipy.optimize import approx_fprime

from pyapprox.surrogates.gaussianprocess.multilevel import (
    MultilevelGaussianProcess)
from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.bayes.markov_chain_monte_carlo import MCMCVariable


class CalibrationGaussianProcess(MultilevelGaussianProcess):
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


class GPCalibrationVariable(MCMCVariable):
    # TODO create new MCMCVariable base class without normalized pdf etc
    # and inherit this from that here and for old MCMCVariable
    def __init__(self, variable, kernel, train_samples, train_values,
                 algorithm="metropolis", **gp_kwargs):
        self.train_samples = train_samples
        self.train_values = train_values
        # estimate hypeprameters using variable.mean()
        self.gp = CalibrationGaussianProcess(kernel, **gp_kwargs)
        self.gp.set_data(train_samples, train_values,
                         variable.get_statistics("mean")*0)
        self.gp.fit()
        # fit overwrite gp.kernel_ with gp.kernel so set length_bounds
        # of both to zero
        self.gp.kernel_.length_scale_bounds = "fixed"
        self.gp.kernel_.rho_bounds = "fixed"
        self.gp.kernel.length_scale_bounds = "fixed"
        self.gp.kernel.rho_bounds = "fixed"

        loglike = self.loglike_calibration_params
        super().__init__(variable, loglike, algorithm, loglike_grad=True,
                         burn_fraction=0.1, njobs=1)

    def _loglike_theta(self, theta):
        assert self.gp.kernel_.length_scale_bounds == "fixed"
        assert self.gp.kernel_.rho_bounds == "fixed"
        assert theta.ndim == 1
        self.gp.set_data(
            self.train_samples, self.train_values, theta[:, None])
        self.gp.fit()
        val = self.gp.log_marginal_likelihood(
            self.gp.kernel_.theta, clone_kernel=False)
        return val

    def _loglike_theta_with_grad(self, theta, jac):
        val = self._loglike_theta(theta)
        if not jac:
            return val
        # Warning map point is very sensitive to finite difference
        # step size. TODO compute gradients analytically
        grad = approx_fprime(theta, self._loglike_theta, 1e-10)
        return val, grad

    def loglike_calibration_params(self, theta, jac=False):
        assert theta.ndim == 2 and theta.shape[1] == 1
        return self._loglike_theta_with_grad(theta[:, 0], jac)

    def loglike_calibration_and_hyperparams(self, zz, jac=False):
        assert zz.ndim == 2 and zz.shape[1] == 1
        nlengthscales = self.gp.nmodels*self.gp.nvars
        nrho = self.gp.nmodels-1
        nhyperparams = nlengthscales+nrho
        assert zz.shape[0] == nhyperparams+self._variable.num_vars()
        self.gp.kernel_.length_scale = zz[:nlengthscales, 0]
        self.gp.kernel_.rho = zz[nlengthscales:nlengthscales+nrho, 0]
        theta = zz[nlengthscales+nrho:, 0]
        return self._loglike_theta_with_grad(theta, jac)

    def __str__(self):
        string = "GPCalibrationVariable with prior:\n"
        string += self._variable.__str__()
        return string
