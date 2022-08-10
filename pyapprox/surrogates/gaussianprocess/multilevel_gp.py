import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from pyapprox.surrogates.gaussianprocess.kernels import (
    MultilevelGPKernel, ConstantKernel)
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GaussianProcess
)


class MultilevelGP(GaussianProcessRegressor):
    def __init__(self, kernel, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None, normalize_y=False):
        if type(kernel) != MultilevelGPKernel:
            raise ValueError("Multilevel Kernel must be provided")
        super().__init__(
            kernel=kernel, alpha=alpha,
            optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y, copy_X_train=copy_X_train,
            random_state=random_state)
        self._samples = None
        self._values = None

    def set_data(self, samples, values):
        self.nmodels = len(samples)
        assert len(values) == self.nmodels
        self._samples = samples
        self._values = values
        assert samples[0].ndim == 2
        assert self._values[0].ndim == 2
        assert self._values[0].shape[1] == 1
        self.nvars = samples[0].shape[0]
        for ii in range(1, self.nmodels):
            assert samples[ii].ndim == 2
            assert samples[ii].shape[0] == self.nvars
            assert self._values[ii].ndim == 2
            assert self._values[ii].shape[1] == 1
            assert self._values[ii].shape[0] == samples[ii].shape[1]

    def fit(self):
        XX_train = np.hstack(self._samples).T
        YY_train = np.vstack(self._values)
        super().fit(XX_train, YY_train)

    def __call__(self, XX_test, return_std=False, return_cov=False,
                 model_eval_id=None):
        if model_eval_id is None:
            model_eval_id = self.nmodels-1
        self.kernel_.model_eval_id = model_eval_id
        result = self.predict(XX_test.T, return_std, return_cov)
        if type(result) != tuple:
            return result
        # when returning prior stdev covariance then must reshape vals
        if result[0].ndim == 1:
            result = [result[0][:, None]] + [r for r in result[1:]]
            result = tuple(result)
        return result

    def plot_1d(self, num_XX_test, bounds,
                ax=None, num_stdev=2, plt_kwargs={}, fill_kwargs={},
                prior_fill_kwargs=None, model_eval_id=None):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
        XX_test = np.linspace(bounds[0], bounds[1], num_XX_test)[None, :]
        # return_std=True does not work for gradient enhanced krigging
        # gp_mean, gp_std = predict(XX_test,return_std=True)
        gp_mean, gp_std = self(
            XX_test, return_std=True, model_eval_id=model_eval_id)
        gp_mean = gp_mean[:, 0]
        if prior_fill_kwargs is not None:
            prior_std = np.sqrt(self.kernel_.diag(XX_test.T, model_eval_id))
            ax.fill_between(
                XX_test[0, :], self._y_train_mean-num_stdev*prior_std,
                self._y_train_mean+num_stdev*prior_std, **prior_fill_kwargs)
        ax.plot(XX_test[0, :], gp_mean, **plt_kwargs)
        ax.fill_between(
            XX_test[0, :], gp_mean-num_stdev*gp_std, gp_mean+num_stdev*gp_std,
            **fill_kwargs)
        return ax


class SequentialMultiLevelGP(MultilevelGP):
    def __init__(self, kernels, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None):
        self._raw_kernels = kernels
        self._alpha = alpha
        self._n_restarts_optimizer = n_restarts_optimizer
        self._copy_X_train = copy_X_train
        self._random_state = random_state

        self._gps = None
        self.rho = None

    def _get_kernel(self, kk):
        from pyapprox.surrogates.gaussianprocess.kernels import SequentialMultiLevelKernel
        kernels = []
        for ii in range(kk):
            self._raw_kernels[ii].length_scale_bounds = "fixed"
            if kk > 1:
                print("rho", ii, kk, self.rho[ii:kk])
                kernels.append(self._raw_kernels[ii]*np.prod(self.rho[ii:kk])**2)
            else:
                kernels.append(self._raw_kernels[ii])
        kernels.append(self._raw_kernels[kk])
        kernel = SequentialMultiLevelKernel(
            self._raw_kernels[:kk+1], np.hstack((
                self._raw_kernels[kk].length_scale, [1])),
            self._raw_kernels[kk].length_scale_bounds)
        return kernel

    def fit(self):
        nmodels = len(self._samples)
        self.rho = [1 for ii in range(nmodels-1)]
        self._gps = []
        lf_values = 0
        for ii in range(nmodels):
            print("####", ii)
            if ii > 1:
                lf_values += self.rho[ii-1]*lf_values + self._gps[ii-1](
                    self._samples[ii])
            elif ii == 1:
                lf_values = self._gps[ii-1](self._samples[ii])
            gp = SequentialMultilevelGaussianProcess(
                lf_values,
                kernel=self._get_kernel(ii),
                n_restarts_optimizer=self._n_restarts_optimizer,
                alpha=self._alpha, random_state=self._random_state,
                copy_X_train=self._copy_X_train)
            gp.fit(self._samples[ii], self._values[ii])
            print(ii, gp.kernel_)
            if ii > 0:
                print(gp.kernel.length_scale)
                self.rho[ii-1] = gp.kernel_.length_scale[-1]
            self._gps.append(gp)

    def __call__(self, samples, model_idx=None):
        nmodels = len(self._gps)
        means, variances = [], []
        if model_idx is None:
            model_idx = [nmodels-1]
        ml_mean, ml_var = 0, 0
        for ii in range(nmodels):
            discrepancy, std = self._gps[ii](samples, return_std=True)
            if ii > 0:
                ml_mean = self.rho[ii-1]*ml_mean + discrepancy
                ml_var = self.rho[ii-1]**2*ml_var + std**2
            else:
                ml_mean = discrepancy
                ml_var = std**2
            if ml_mean.ndim == 1:
                ml_mean = ml_mean[:, None]
            means.append(ml_mean)
            variances.append(ml_var)
        if len(model_idx) == 1:
            return (means[model_idx[0]],
                    np.sqrt(variances[model_idx[0]]).squeeze())
        return ([means[idx] for idx in model_idx],
                [np.sqrt(variances[idx]).squeeze() for idx in model_idx])


class SequentialMultilevelGaussianProcess(GaussianProcess):
    def __init__(self, lf_values, **kwargs):
        super().__init__(**kwargs)
        self.lf_values = lf_values
        print('k', self.kernel, self.kernel.length_scale_bounds)

    def _log_marginal_likelihood(self, theta):
        rho = theta[-1]
        shift = rho*self.lf_values
        self.y_train_ = self.y_train_-shift
        val = super().log_marginal_likelihood(theta, clone_kernel=False)
        return val

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        val = self._log_marginal_likelihood(theta)
        if not eval_gradient:
            return val
        from scipy.optimize import approx_fprime
        grad = approx_fprime(
            theta, self._log_marginal_likelihood, 1e-10)
        return val, grad
