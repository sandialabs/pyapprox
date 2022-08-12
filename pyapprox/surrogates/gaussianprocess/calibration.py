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
                 algorithm="metropolis", loglike_grad=False, **gp_kwargs):
        self.train_samples = train_samples
        self.train_values = train_values

        if loglike_grad is False:
            loglike_grad = None
        # else: use finite difference

        loglike = self.loglike_calibration_params
        super().__init__(
            variable, loglike, algorithm, loglike_grad=loglike_grad,
            burn_fraction=0.1, njobs=1)

        self._set_hyperparams(kernel, **gp_kwargs)

    def _fix_hyperparameters(self):
        if hasattr(self.gp, "kernel_"):
            self.gp.kernel_.length_scale_bounds = "fixed"
            self.gp.kernel_.rho_bounds = "fixed"
        self.gp.kernel.length_scale_bounds = "fixed"
        self.gp.kernel.rho_bounds = "fixed"

    def _unfix_hyperparameters(self):
        if hasattr(self.gp, "kernel_"):
            self.gp.kernel_.length_scale_bounds = (1e-5, 1e1)
            self.gp.kernel_.rho_bounds = (1e-5, 1e1)
        self.gp.kernel.length_scale_bounds = (1e-5, 1e1)
        self.gp.kernel.rho_bounds = (1e-5, 1e1)

    def _set_hyperparams(self, kernel, **gp_kwargs):
        # estimate hypeprameters using variable.mean()
        init_theta = self._variable.get_statistics("mean")
        self.gp = CalibrationGaussianProcess(kernel, **gp_kwargs)
        self.gp.set_data(self.train_samples, self.train_values,
                         init_theta)
        self.gp.fit()
        self._fix_hyperparameters()

        # # if not calling fit must add the following line and comment
        # out _fix_hyperparameters above
        # self.gp.kernel_ = self.gp.kernel

        # # fit overwrite gp.kernel_ with gp.kernel so set length_bounds
        # # of both to zero
        # hyperparams_bounds = self.gp.kernel.bounds
        # # bounds must be obtained before length_scale is fixed
        # self._fix_hyperparameters()

        # #init_hyperparams = np.hstack((self.gp.kernel_.length_scale.copy(),
        # #                              self.gp.kernel_.rho.copy()))
        # init_hyperparams = np.exp(hyperparams_bounds.sum(axis=1)/2)
        # print(init_hyperparams, "init H")

        # init_guess = np.hstack((init_hyperparams, init_theta[:, 0]))
        # import scipy
        # theta_bounds = self._variable.get_statistics("interval", alpha=1)
        # bounds = np.vstack((np.exp(hyperparams_bounds), theta_bounds))
        # res = scipy.optimize.minimize(
        #     lambda x: self.loglike_calibration_and_hyperparams(x[:, None]),
        #     init_guess+np.random.normal(0, 1e-1),
        #     method="L-BFGS-B", jac=False,
        #     # method="Powell",
        #     options={"disp": True, "eps": 1e-3},
        #     bounds=bounds)
        # hyperparams = res.x[:hyperparams_bounds.shape[0]]
        # # hyperparameters must be temporarily unfixed to set kernel_.theta_
        # self._unfix_hyperparameters()
        # self.gp.kernel_.theta = np.log(hyperparams)
        # self._fix_hyperparameters()
        # self.MAP = res.x[hyperparams_bounds.shape[0]:]
        # print(self.MAP)
        # print(res.x)
        # assert False

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
        grad = approx_fprime(theta, self._loglike_theta, 1e-8)
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
        val = self._loglike_theta_with_grad(theta, jac)
        print(zz, val)
        return val

    def __str__(self):
        string = "GPCalibrationVariable with prior:\n"
        string += self._variable.__str__()
        return string

    def _plot_loglikelihood_cross_section(self, ax, bounds):
        # nominal values are the current values in self.kernel_
        # TODO generalize. Currently only works for varying 1 rho and 1 theta
        from pyapprox.util.visualization import get_meshgrid_function_data, plt
        from pyapprox.interface.wrappers import (
            evaluate_1darray_function_on_2d_array)
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        from functools import partial
        hyperparams = np.hstack((self.gp.kernel_.length_scale.copy(),
                                 self.gp.kernel_.rho.copy()))

        def plotfun_1d_array(xx):
            zz = np.hstack((hyperparams, xx[1]))
            zz[-2] = xx[0]
            return self.loglike_calibration_and_hyperparams(
                zz[:, None])
        plotfun = partial(
            evaluate_1darray_function_on_2d_array, plotfun_1d_array)

        X, Y, Z = get_meshgrid_function_data(plotfun, bounds, 30)
        im = plt.contourf(
            X, Y, Z,  levels=np.linspace(Z.min(), Z.max(), 20))
        plt.colorbar(im, ax=ax)
        return ax


# TODO: obs_model kernel should not have length scales associated
# with random variables. This should not effect result but
# requires wasted optimization effort
