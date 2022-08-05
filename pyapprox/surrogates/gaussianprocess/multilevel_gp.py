import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    StationaryKernelMixin, NormalizedKernelMixin, Kernel, Hyperparameter,
    _approx_fprime
)

from pyapprox.surrogates.gaussianprocess.gradient_enhanced_gp import (
    plot_gp_1d, kernel_ff, get_gp_samples_kernel
)
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    GaussianProcess
)


def multilevel_diagonal_covariance_block(XX1, XX2, hyperparams, mm):
    """
    Models assumed to be ordered by increasing fidelity.
    Warning this is different to other multifidelity kriging approaches


    Parameters
    ==========
    hyperparams : np.ndarray (nvars*nmodels+nmodels-1)
        The nvars length_scales of each kernel and then the correlation (rho)
        of each model (except lowest fidelity model)
    """

    nvars = XX1.shape[1]
    nmodels = (len(hyperparams)+1)//(nvars+1)
    length_scales = np.asarray(hyperparams[:(mm+1)*nvars])
    rho = np.asarray(hyperparams[nmodels*nvars:nmodels*nvars+mm])
    K = 0
    for kk in range(mm):
        K += np.prod(rho[:kk+1])**2*kernel_ff(
            XX1, XX2, length_scales[nvars*kk:nvars*(kk+1)])
    K += kernel_ff(XX1, XX2, length_scales[nvars*mm:nvars*(mm+1)])
    return K


def multilevel_off_diagonal_covariance_block(XXmm, XXnn, hyperparams, mm, nn):
    """
    Parameters
    ----------

    XXmm : np.ndarray (nsamples_mm,nvars)
        The samples of the mm the model

    XXnn : np.ndarray (nsamples_nn,nvars)
        The samples of the nn the model
    """
    assert mm < nn
    nvars = XXmm.shape[1]
    nmodels = (len(hyperparams)+1)//(nvars+1)
    K = multilevel_diagonal_covariance_block(XXmm, XXnn, hyperparams, mm)
    rho = np.asarray(hyperparams[nmodels*nvars:nmodels*nvars+nn])
    K *= np.prod(rho[mm:nn])
    return K


def unpack_samples(XX, nsamples_per_model):
    samples = []
    lb, ub = 0, 0
    for nn in nsamples_per_model:
        lb = ub
        ub += nn
        samples.append(XX[lb:ub, :])
    return samples


def full_multilevel_kernel(
        XX1, hyperparams, nsamples_per_model, hf_only=False):
    nrows = XX1.shape[0]
    nmodels = len(nsamples_per_model)
    if hf_only:
        K = multilevel_diagonal_covariance_block(
            XX1, XX1, hyperparams, nmodels-1)
        return K
    assert np.sum(nsamples_per_model) == XX1.shape[0]

    samples = unpack_samples(XX1, nsamples_per_model)
    K = np.zeros((nrows, nrows), dtype=float)
    lb1, ub1 = 0, 0
    for mm in range(nmodels):
        lb1 = ub1
        ub1 += nsamples_per_model[mm]
        lb2, ub2 = lb1, ub1
        K_block = multilevel_diagonal_covariance_block(
            samples[mm], samples[mm], hyperparams, mm)
        K[lb1:ub1, lb2:ub2] = K_block
        for nn in range(mm+1, nmodels):
            lb2 = ub2
            ub2 += nsamples_per_model[nn]
            K[lb1:ub1, lb2:ub2] = multilevel_off_diagonal_covariance_block(
                samples[mm], samples[nn], hyperparams, mm, nn)
            K[lb2:ub2, lb1:ub1] = K[lb1:ub1, lb2:ub2].T
    return K


def multilevel_kernel_for_prediction(XX1, train_samples_mm, hyperparams,
                                     nsamples_per_model, mm):
    nvars = XX1.shape[1]
    nmodels = len(nsamples_per_model)
    length_scales = np.asarray(hyperparams[:(mm+1)*nvars])
    rho = np.asarray(hyperparams[nmodels*nvars:nmodels*nvars+nmodels])
    lb, ub = mm*nvars, (mm+1)*nvars
    K = np.prod(rho[mm:])*kernel_ff(
        XX1, train_samples_mm, length_scales[lb:ub])
    if mm == 0:
        return K
    K += rho[mm-1]*multilevel_kernel_for_prediction(
        XX1, train_samples_mm, hyperparams, nsamples_per_model, mm-1)
    return K


def full_multilevel_kernel_for_prediction(
        XX1, XX2, hyperparams, nsamples_per_model):
    """
    XX2 : np.ndarray (nsamples_per_model.sum(),nvars)
        Training samples for all models
    """
    train_samples = unpack_samples(XX2, nsamples_per_model)
    nmodels = len(train_samples)
    K = []
    for mm in range(nmodels):
        k_block = multilevel_kernel_for_prediction(
            XX1, train_samples[mm], hyperparams, nsamples_per_model, mm)
        K.append(k_block)
    K = np.hstack(K)
    return K


class MultilevelGPKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, nvars, nsamples_per_model, length_scale=[1.0],
                 length_scale_bounds=(1e-5, 1e5)):
        """
        Parameters
        ----------
        length_scale : np.ndarray ((nmodels+1)*nvars-1)
            The length scales of each model kernel and the correlation between
            consecutive models (lowest fidelity does not have a rho)
        """
        self.nmodels = len(nsamples_per_model)
        self.nvars = nvars
        self.nsamples_per_model = nsamples_per_model
        self.return_code = 'full'

        assert len(length_scale) == self.nmodels*nvars+self.nmodels-1
        self.length_scale = np.atleast_1d(length_scale)
        if length_scale_bounds != 'fixed':
            assert len(length_scale_bounds) == self.nmodels * \
                nvars+self.nmodels-1
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, XX1, XX2=None, eval_gradient=False):
        """Return the kernel k(XX1, XX2) and optionally its gradient.

        Parameters
        ----------
        XX1 : array, shape (n_samples_XX1, n_features)
            Left argument of the returned kernel k(XX1, XX2)

        XX2 : array, shape (n_samples_XX2, n_features), (optional, default=None)
            Right argument of the returned kernel k(XX1, XX2). If None,
            k(XX1, XX1) is evaluated instead.

        Returns
        -------
        K : array, shape (n_samples_XX1, n_samples_XX2)
            Kernel k(XX1, XX2)

        K_gradient : array (opt.), shape (n_samples_XX1, n_samples_XX1, n_dims)
            The gradient of the kernel k(XX1, XX1) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        XX1 = np.atleast_2d(XX1)
        hyperparams = np.squeeze(self.length_scale).astype(float)
        if XX2 is None:
            K = full_multilevel_kernel(
                XX1, hyperparams, self.nsamples_per_model,
                self.return_code != 'full')
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when XX2 is None.")
            K = full_multilevel_kernel_for_prediction(
                XX1, XX2, hyperparams, self.nsamples_per_model)
        if not eval_gradient:
            return K

        if self.hyperparameter_length_scale.fixed:
            # Hyperparameter l kept fixed
            length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))
        else:
            # approximate gradient numerically
            def f(gamma):  # helper function
                return full_multilevel_kernel(
                    XX1, gamma, self.nsamples_per_model)
            length_scale = np.atleast_1d(self.length_scale)
            length_scale_gradient = _approx_fprime(length_scale, f, 1e-8)

        return K, length_scale_gradient

    def diag(self, X):
        """
        Need to overide this function as base kernel.diag methods only returns
        1. TODO add this to gradient enhanced kriging then I can remove
        covariance hack from plot_gp_1d
        TODO make this function more efficient by directly evaluating diagonal
        only
        """
        return np.diag(self(X)).copy()

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __repr__(self):
        self.length_scale = np.atleast_1d(self.length_scale)
        return "{0}(length_scale=[{1}])".format(
            self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))


class MultilevelGP(GaussianProcessRegressor):
    def __init__(self, kernel, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None):
        normalize_y = False
        super(MultilevelGP, self).__init__(
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
        # YY_train = np.hstack(self._values)
        YY_train = np.vstack(self._values)
        super().fit(XX_train, YY_train)

    def __call__(self, XX_test, return_std=False, return_cov=False):
        gp_kernel = get_gp_samples_kernel(self)
        return_code = gp_kernel.return_code
        gp_kernel.return_code = 'values'
        result = super().predict(XX_test.T, return_std, return_cov)
        gp_kernel.return_code = return_code
        if type(result) != tuple:
            return result
        # when returning prior stdev covariance then must reshape vals
        if result[0].ndim == 1:
            result = [result[0][:, None]] + [r for r in result[1:]]
            result = tuple(result)
        return result

    def plot_1d(self, num_XX_test, bounds, axs, num_stdev=2, function=None,
                gp_label=None, function_label=None, model_id=None):
        assert self.nvars == 1
        assert self.nvars == len(bounds)//2
        if model_id is None:
            model_id = len(self._samples)-1

        # sklearn requires samples to (nsamples, nvars)
        XX_train = self._samples[model_id].T
        YY_train = self._values[model_id]
        plot_gp_1d(
            axs, self.predict, num_XX_test, bounds, XX_train, YY_train,
            num_stdev, function, gp_label=gp_label,
            function_label=function_label)


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
        self._rho = None

    def fit(self, rho):
        nmodels = len(self._samples)
        self._rho = rho
        self._kernels = [self._raw_kernels[0]]
        for ii in range(1, nmodels):
            self._kernels.append(
                self._rho[ii-1]**2*self._kernels[ii-1]+self._raw_kernels[ii])

        self._gps = []
        for ii in range(nmodels):
            # length_scale=.1, length_scale_bounds='fixed')
            # gp_kernel += WhiteKernel( # optimize gp noise
            #    noise_level=noise_level,
            #    noise_level_bounds=noise_level_bounds)
            gp = GaussianProcess(
                kernel=self._kernels[ii],
                n_restarts_optimizer=self._n_restarts_optimizer,
                alpha=self._alpha, random_state=self._random_state,
                copy_X_train=self._copy_X_train)
            if ii > 0:
                shift = rho[ii-1]*self._gps[ii-1](self._samples[ii])
                print(self._values[ii]-shift)
            else:
                shift = 0
            gp.fit(self._samples[ii], self._values[ii]-shift)
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
                ml_mean = self._rho[ii-1]*ml_mean + discrepancy
                ml_var = self._rho[ii-1]**2*ml_var + std**2
            else:
                ml_mean = discrepancy
                ml_var = std**2
            means.append(ml_mean)
            variances.append(ml_var)
        if len(model_idx) == 1:
            return (means[model_idx[0]],
                    np.sqrt(variances[model_idx[0]]).squeeze())
        return ([means[idx] for idx in model_idx],
                [np.sqrt(variances[idx]).squeeze() for idx in model_idx])
