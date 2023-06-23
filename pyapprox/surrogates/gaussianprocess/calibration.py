import numpy as np
import copy
from scipy.optimize import approx_fprime, minimize

from pyapprox.surrogates.gaussianprocess.multilevel import (
    MultifidelityGaussianProcess)
from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.bayes.metropolis import MetropolisMCMCVariable


class CalibrationGaussianProcess(MultifidelityGaussianProcess):
    def set_data(self, samples, values, random_sample):
        # when length_scale bounds is fixed then the V2(D2) block in Kennedys
        # paper should always be the same regardless of value of random_sample
        # Currently I have extra length scales for V2 kernel, i.e. for
        # random_sample dimensions. This will not effect answer given other
        # increase size of optimization problem
        samples_copy = copy.deepcopy(samples)
        samples_copy[-1] = get_all_sample_combinations(
            samples[-1], random_sample)
        super().set_data(samples_copy, values)

    def plot_1d(self, random_sample, num_XX_test, bounds,
                ax=None, num_stdev=2, plt_kwargs={}, fill_kwargs={},
                prior_fill_kwargs=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[None, :]
        # return_std=True does not work for gradient enhanced krigging
        # gp_mean, gp_std = predict(XX_test,return_std=True)
        XX = get_all_sample_combinations(XX_test, random_sample)
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


class GPCalibrationVariable(MetropolisMCMCVariable):
    # TODO create new MCMCVariable base class without normalized pdf etc
    # and inherit this from that here and for old MCMCVariable
    def __init__(self, variable, kernel, train_samples, train_values,
                 algorithm="metropolis", loglike_grad=None, **gp_kwargs):
        self.train_samples = train_samples
        self.train_values = train_values

        # if loglike_grad is False:
        #     loglike_grad = None
        # # else: use finite difference

        self._length_scale_bounds = kernel.length_scale_bounds
        self._rho_bounds = kernel.rho_bounds
        self._sigma_bounds = kernel.sigma_bounds

        loglike = self.loglike_calibration_params
        super().__init__(variable, loglike)

        self.MAP = None
        self._set_hyperparams(kernel, **gp_kwargs)

    def _fix_hyperparameters(self):
        if hasattr(self.gp, "kernel_"):
            self.gp.kernel_.length_scale_bounds = "fixed"
            self.gp.kernel_.rho_bounds = "fixed"
            self.gp.kernel_.sigma_bounds = "fixed"
        self.gp.kernel.length_scale_bounds = "fixed"
        self.gp.kernel.rho_bounds = "fixed"
        self.gp.kernel.sigma_bounds = "fixed"

    def _unfix_hyperparameters(self):
        if hasattr(self.gp, "kernel_"):
            self.gp.kernel_.length_scale_bounds = self._length_scale_bounds
            self.gp.kernel_.rho_bounds = self._rho_bounds
            self.gp.kernel_.sigma_bounds = self._sigma_bounds
        self.gp.kernel.length_scale_bounds = self._length_scale_bounds
        self.gp.kernel.rho_bounds = self._rho_bounds
        self.gp.kernel.sigma_bounds = self._sigma_bounds

    def _set_hyperparams(self, kernel, **gp_kwargs):
        # estimate hypeprameters using variable.mean()
        self.gp = CalibrationGaussianProcess(kernel, **gp_kwargs)
        # init_sample = self._variable.get_statistics("mean")
        # self.gp.set_data(self.train_samples, self.train_values,
        #                 init_sample)
        # self.gp.fit()
        # self._fix_hyperparameters()
        self.MAP = self.maximum_aposteriori_point()
        self.gp.kernel = self.gp.kernel_

    def maximum_aposteriori_point(self, init_sample=None):
        if self.MAP is not None:
            return self.MAP

        if init_sample is None:
            init_sample = self._variable.get_statistics("mean")
        # fit overwrite gp.kernel_ with gp.kernel so set length_bounds
        # of both to zero
        self.gp.nmodels = len(self.train_samples)
        self.gp.nvars = self.train_samples[0].shape[0]
        self.gp.kernel_ = self.gp.kernel
        self._unfix_hyperparameters()
        hyperparams_bounds = self.gp.kernel.bounds
        # bounds must be obtained before length_scale is fixed
        self._fix_hyperparameters()
        init_hyperparams = np.exp(hyperparams_bounds.sum(axis=1)/2)
        init_guess = np.hstack((init_hyperparams, init_sample[:, 0]))
        sample_bounds = self._variable.get_statistics("interval", 1)
        bounds = np.vstack((np.exp(hyperparams_bounds), sample_bounds))
        # fd_eps = 2*np.sqrt(np.finfo(float).eps)
        # fd_eps = 1e-7
        # USE powell because do not want to use finite difference gradients
        # which are very sensitive to finite difference size
        # TODO compute gradient analytically
        res = minimize(
            lambda x: self.negloglike_calibration_and_hyperparams(x[:, None]),
            init_guess,  # +np.random.normal(0, 1e-1),
            # method="L-BFGS-B", jac=False,
            # options={"disp": False, "eps": fd_eps},
            method="Powell",
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
            bounds=bounds)
        if not res.success:
            print(res)
            raise RuntimeError("Optimization failed.")
        hyperparams = res.x[:hyperparams_bounds.shape[0]]
        # hyperparameters must be temporarily unfixed to set kernel_.theta_
        self._unfix_hyperparameters()
        self.gp.kernel_.theta = np.log(hyperparams)
        res.x[hyperparams_bounds.shape[0]:]
        self._fix_hyperparameters()
        return res.x[hyperparams_bounds.shape[0]:, None]

    def _loglike_sample(self, sample):
        assert self.gp.kernel.length_scale_bounds == "fixed"
        assert self.gp.kernel.rho_bounds == "fixed"
        assert self.gp.kernel_.length_scale_bounds == "fixed"
        assert self.gp.kernel_.rho_bounds == "fixed"
        assert sample.ndim == 1
        self.gp.set_data(
            self.train_samples, self.train_values, sample[:, None])
        self.gp.fit()
        val = self.gp.log_marginal_likelihood(
            self.gp.kernel_.theta, clone_kernel=False)
        return val

    def _loglike_sample_with_grad(self, sample, return_grad):
        val = self._loglike_sample(sample)
        if not return_grad:
            return val
        # Warning map point is very sensitive to finite difference
        # step size. TODO compute gradients analytically
        grad = approx_fprime(sample, self._loglike_sample, 1e-8)
        return val, grad

    def loglike_calibration_params(self, sample, return_grad=False):
        assert sample.ndim == 2 and sample.shape[1] == 1
        return self._loglike_sample_with_grad(sample[:, 0], return_grad)

    def negloglike_calibration_and_hyperparams(self, zz):
        assert zz.ndim == 2 and zz.shape[1] == 1
        nlengthscales = self.gp.nmodels*self.gp.nvars
        nrho = self.gp.nmodels-1
        if self._sigma_bounds != "fixed":
            nsigma = self.gp.nmodels
        else:
            nsigma = 0
        nhyperparams = nlengthscales+nrho+nsigma
        assert zz.shape[0] == nhyperparams+self._variable.num_vars()
        self.gp.kernel_.length_scale = zz[:nlengthscales, 0]
        self.gp.kernel_.rho = zz[nlengthscales:nlengthscales+nrho, 0]
        self.gp.kernel.length_scale = zz[:nlengthscales, 0]
        self.gp.kernel.rho = zz[nlengthscales:nlengthscales+nrho, 0]

        if self._sigma_bounds != "fixed":
            self.gp.kernel_.sigma = (
                zz[nlengthscales+nrho:nlengthscales+nrho+nsigma, 0])
            self.gp.kernel.sigma = (
                zz[nlengthscales+nrho:nlengthscales+nrho+nsigma, 0])
        
        sample = zz[nhyperparams:, 0]
        val = self._loglike_sample_with_grad(sample, False)
        val += self._variable.pdf(sample[:, None], log=True)[0, 0]
        return -val

    def __str__(self):
        string = "GPCalibrationVariable with prior:\n"
        string += self._variable.__str__()
        return string
    
    def _plot_negloglikelihood_cross_section(self, ax, bounds):
        # nominal values are the current values in self.kernel_
        # TODO generalize. Currently only works for varying 1 rho and 1 sample
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
            return self.negloglike_calibration_and_hyperparams(
                zz[:, None])
        plotfun = partial(
            evaluate_1darray_function_on_2d_array, plotfun_1d_array)

        X, Y, Z = get_meshgrid_function_data(plotfun, bounds, 30)
        ub = min(Z.min()+0.2*abs(Z.min()), Z.max())
        im = plt.contourf(
            X, Y, Z,  levels=np.linspace(Z.min(), ub, 20))
        plt.colorbar(im, ax=ax)
        return ax


# TODO: obs_model kernel should not have length scales associated
# with random variables. This should not effect result but
# requires wasted optimization effort
