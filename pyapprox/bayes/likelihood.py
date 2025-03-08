from abc import abstractmethod

import numpy as np
from pyapprox.util.pya_numba import njit

from pyapprox.util.linalg import (
    inverse_of_cholesky_factor,
    log_determinant_from_cholesky_factor,
    diag_of_mat_mat_product,
)
from pyapprox.interface.model import Model
from pyapprox.variables.joint import JointVariable
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin


class LogLikelihood(Model):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend=backend)
        self._obs = None
        self._nobs = None

    def _set_tile_obs(self, tile_obs: bool):
        self._tile_obs = tile_obs

    def _parse_obs(self, obs: Array, many_pred_obs: Array) -> Array:
        if not hasattr(self, "_loglike_const"):
            raise RuntimeError("must call _setup()")
        if not hasattr(self, "_tile_obs"):
            raise RuntimeError("must call _set_tile_obs()")
        if self._tile_obs:
            # stack vals for each obs vertically
            return (
                self._bkd.repeat(self._obs, many_pred_obs.shape[1], axis=1),
                self._bkd.tile(many_pred_obs, (1, self._obs.shape[1])),
            )
        if many_pred_obs.shape != self._obs.shape:
            msg = "many_pred_obs shape does not match self._obs shape"
            raise ValueError(msg)
        return (self._obs, many_pred_obs)

    def nobs(self) -> int:
        return self._nobs

    def nqoi(self) -> int:
        return 1

    def set_observations(self, obs: Array):
        """
        Parameters
        ----------
        obs : np.ndarray (nobs, 1)
            The observations
        """
        if obs.ndim != 2 or obs.shape[0] != self.nobs():
            raise ValueError("obs has the wrong shape {0}".format(obs.shape))
        self._obs = obs

    @abstractmethod
    def _loglike(self, many_pred_obs: Array) -> Array:
        raise NotImplementedError

    def _check_many_pred_obs(self, many_pred_obs: Array):
        if many_pred_obs.shape[0] != self.nobs():
            raise ValueError(
                "many_pred_obs has the wrong shape {0}".format(
                    many_pred_obs.shape
                )
                + " nobs is {0}".format(self.nobs())
            )

    def _loglike_from_pred_obs(self, many_pred_obs: Array) -> Array:
        self._check_many_pred_obs(many_pred_obs)
        return self._loglike(many_pred_obs)

    @abstractmethod
    def _make_noisy(self, noiseless_obs: Array, noise, Array):
        raise NotImplementedError


class ModelBasedLogLikelihoodMixin:
    def model(self) -> Model:
        return self._model

    def nvars(self) -> int:
        return self._model.nvars()

    def _values(self, samples: Array):
        many_pred_obs = self._model(samples).T
        self._last_sample = samples[:, -1:]
        self._last_pred_obs = many_pred_obs[:, -1:]
        return self._loglike_from_pred_obs(many_pred_obs)

    def rvs(self, sample: Array) -> Array:
        """Draw a realization from the likelihood"""
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError("samples has the wrong shape")
        noiseless_obs = self._model(sample).T
        noise = self._sample_noise(1)
        return self._make_noisy(noiseless_obs, noise)

    def __repr__(self) -> str:
        return "{0}(nobs={1}, model={2})".format(
            self.__class__.__name__, self.nobs(), self._model
        )


class GaussianLogLikelihood(LogLikelihood):
    """GaussianLikelihood with dense noise covariance matrix"""

    # TODO this code uses a lot of Multivariate Gaussian. Merge
    # MultivariateGaussian(obs, loglike.noise_covariance())

    def set_design_weights(self, design_weights: Array) -> Array:
        self._design_weights = design_weights
        self._noise_chol = self._bkd.cholesky(self._noise_cov)
        self._noise_chol_inv = inverse_of_cholesky_factor(
            self._noise_chol, self._bkd
        )
        self._wnoise_chol_inv = self._noise_chol_inv * self._bkd.sqrt(
            self._design_weights
        )
        # Todo avoid inverting and just apply weights is possible
        self._wnoise_chol = inverse_of_cholesky_factor(
            self._wnoise_chol_inv, self._bkd
        )
        self._log_wnoise_cov_det = log_determinant_from_cholesky_factor(
            self._wnoise_chol, self._bkd
        )
        self._loglike_const = (
            self._nobs * np.log(2 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov: Array):
        if noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]:
            raise ValueError(
                "noise_cov has the wrong shape {0}".format(noise_cov.shape)
            )
        self._noise_cov = noise_cov
        self._nobs = noise_cov.shape[0]
        self._last_L_inv_res = None
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def _make_noisy(self, noiseless_obs: Array, noise: Array) -> Array:
        if noiseless_obs.shape != noise.shape:
            msg = "shapes of noiseless_obs {0} and obs {1} must match".format(
                noiseless_obs.shape, noise.shape
            )
            raise ValueError(msg)
        return noiseless_obs + noise

    def _sample_noise(self, nsamples: int) -> Array:
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self.nobs()))
        ).T
        return self._wnoise_chol @ normal_samples

    def _loglike_many(self, many_obs: Array, many_pred_obs: Array) -> Array:
        residual = many_obs - many_pred_obs
        L_inv_res = self._wnoise_chol_inv @ residual
        vals = diag_of_mat_mat_product(L_inv_res.T, L_inv_res, bkd=self._bkd)
        return (-0.5 * (vals + self._loglike_const))[:, None]

    def _loglike(self, many_pred_obs: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        return self._loglike_many(*self._parse_obs(self._obs, many_pred_obs))

    def noise_covariance(self) -> Array:
        """Return weighted noise covariance"""
        return self._wnoise_chol @ self._wnoise_chol.T


class IndependentGaussianLogLikelihood(GaussianLogLikelihood):
    """
    Independent but not necessarily identically distributed Gaussian noise.
    Evaluating this is faster than GaussianLikelihood if noise covariance
    is a diagonal matrix
    """

    def set_design_weights(self, design_weights: Array):
        self._design_weights = design_weights
        assert self._bkd.all(design_weights >= 0), design_weights
        self._noise_cov_inv_diag = 1 / (self._noise_cov_diag)
        self._wnoise_cov_inv_diag = (
            self._noise_cov_inv_diag * self._design_weights
        )
        self._wnoise_std_diag = self._bkd.sqrt(1 / self._wnoise_cov_inv_diag)
        self._log_wnoise_cov_det = (
            2 * self._bkd.log(self._wnoise_std_diag).sum()
        )
        self._r_noise_cov_inv = None
        self._loglike_const = (
            self._nobs * np.log(2 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov_diag: Array):
        print(noise_cov_diag.shape)
        if noise_cov_diag.ndim != 2 or noise_cov_diag.shape[1] != 1:
            raise ValueError(
                "noise_cov_diag has the wrong shape {0}".format(
                    noise_cov_diag.shape
                )
            )
        self._nobs = noise_cov_diag.shape[0]
        self._noise_cov_diag = noise_cov_diag
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def _loglike_many(self, many_obs, many_pred_obs):
        weighted_residual = self._bkd.sqrt(self._wnoise_cov_inv_diag) * (
            many_obs - many_pred_obs
        )
        vals = (
            -0.5
            * diag_of_mat_mat_product(
                weighted_residual.T, weighted_residual, bkd=self._bkd
            )
            - 0.5 * self._loglike_const
        )[:, None]
        return vals

    def _sample_noise(self, nsamples):
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self.nobs()))
        ).T
        return self._wnoise_std_diag * normal_samples

    def noise_covariance(self):
        return self._bkd.diag(1 / self._wnoise_cov_inv_diag[:, 0])

    def __repr__(self):
        return "{0}(nobs={1}, sigma={2})".format(
            self.__class__.__name__,
            self.nobs(),
            self._noise_std_diag[:, 0],
        )


class ModelBasedGaussianLogLikelihood(
    ModelBasedLogLikelihoodMixin, GaussianLogLikelihood
):
    def __init__(self, model: Model, noise_cov: Array, tile_obs: bool = True):
        super().__init__(backend=model._bkd)
        self._model = model
        self._setup(noise_cov)
        self._set_tile_obs(tile_obs)

    def jacobian_implemented(self) -> bool:
        return self._model.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        if self._last_sample is None or not self._bkd.allclose(
            sample, self._last_sample
        ):
            self._last_sample = sample.copy()
            self._last_pred_obs = self._model(sample).T
        residual = self._obs - self._last_pred_obs
        L_inv_res = self._wnoise_chol_inv @ residual
        return self._bkd.multidot(
            (
                L_inv_res.T,
                self._wnoise_chol_inv,
                self._model.jacobian(sample),
            )
        )

    # TODO implement apply_hessian using same idea as used in local
    # optimaility criteria in expdesign


class ModelBasedIndependentGaussianLogLikelihood(
    ModelBasedLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    def __init__(
        self, model: Model, noise_cov_diag: Array, tile_obs: bool = True
    ):
        super().__init__(backend=model._bkd)
        self._model = model
        self._setup(noise_cov_diag)
        self._set_tile_obs(tile_obs)

    def jacobian_implemented(self) -> bool:
        return self._model.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        if self._last_sample is None or not self._bkd.allclose(
            sample, self._last_sample
        ):
            self._last_sample = sample.copy()
            self._last_pred_obs = self._model(sample).T
        residual = self._obs - self._last_pred_obs
        tmp = self._wnoise_cov_inv_diag * residual
        return tmp.T @ self._model.jacobian(sample)


class WeightBasedLogLikelihoodMixin:
    def nvars(self) -> int:
        return self._model.nvars()


class WeightBasedGaussianLogLikelihood(
    WeightBasedLogLikelihoodMixin, GaussianLogLikelihood
):
    pass


class IndependentExponentialLogLikelihood(LogLikelihood):
    r"""
    p(y|z)=\lambda\exp(-\lambda (y-f(z)))
    lambda = scale
    log p(y|z) = log(\lambda)-\lambda*(y-f(z))
    """

    def set_design_weights(self, design_weights):
        self._design_weights = design_weights
        self._wnoise_scale_diag = self._noise_scale_diag * self._design_weights
        self._loglike_const = self._bkd.log(self._wnoise_scale_diag).sum()

    def _setup(self, noise_scale_diag):
        if noise_scale_diag.ndim != 2 or noise_scale_diag.shape[1] != 1:
            raise ValueError(
                "noise_scale_diag has the wrong shape {0}".format(
                    noise_scale_diag.shape
                )
            )
        self._nobs = noise_scale_diag.shape[0]
        self._noise_scale_diag = noise_scale_diag
        self._noise_log_scale_diag = self._bkd.log(self._noise_scale_diag)
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def _parse_obs(self, obs, many_pred_obs):
        if self._tile_obs:
            # stack vals for each obs vertically
            return (
                self._bkd.repeat(self._obs, many_pred_obs.shape[1], axis=1),
                self._bkd.tile(many_pred_obs, (1, self._obs.shape[1])),
            )
        if many_pred_obs.shape != self._obs.shape:
            msg = "many_pred_obs shape does not match self._obs shape"
            raise ValueError(msg)
        return (self._obs, many_pred_obs)

    def _loglike(self, many_pred_obs):
        return self._loglike_many(*self._parse_obs(self._obs, many_pred_obs))

    def _loglike_many(self, many_obs, many_pred_obs):
        return (
            self._loglike_const
            - (self._wnoise_scale_diag * (many_obs - many_pred_obs)).sum(
                axis=0
            )[:, None]
        )

    def _make_noisy(self, noiseless_obs, noise):
        if noiseless_obs.shape != noise.shape:
            msg = "shapes of noiseless_obs {0} and obs {1} must match".format(
                noiseless_obs.shape, noise.shape
            )
            raise ValueError(msg)
        return noiseless_obs + noise

    def _sample_noise(self, nsamples):
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        exponential_samples = self._bkd.asarray(
            np.random.exponential(1, (nsamples, self.nobs()))
        ).T
        return 1 / self._wnoise_scale_diag * exponential_samples


class ModelBasedIndependentExponentialLogLikelihood(
    ModelBasedLogLikelihoodMixin, IndependentExponentialLogLikelihood
):
    def __init__(
        self, model: Model, noise_scale_diag: Array, tile_obs: bool = True
    ):
        super().__init__(backend=model._bkd)
        self._model = model
        self._setup(noise_scale_diag)
        self._set_tile_obs(tile_obs)


class LogUnormalizedPosterior(Model):
    def __init__(self, loglike: LogLikelihood, prior: JointVariable):
        self._loglike = loglike
        self._prior = prior
        super().__init__(loglike._bkd)

    def nqoi(self) -> int:
        return 1

    def nvars(self) -> int:
        return self._loglike.nvars()

    def jacobian_implemented(self) -> bool:
        return (
            self._prior.pdf_jacobian_implemented()
            and self._loglike.jacobian_implemented()
        )

    def hessian_implemented(self) -> bool:
        print(
            self._prior.pdf_hessian_implemented(),
            self._loglike.hessian_implemented(),
        )
        return (
            self._prior.pdf_hessian_implemented()
            and self._loglike.hessian_implemented()
        )

    def _values(self, samples: Array) -> Array:
        return self._loglike(samples) + self._prior._log_pdf(samples)

    def _jacobian(self, sample: Array) -> Array:
        return self._loglike.jacobian(sample) + self._prior.log_pdf_jacobian(
            sample
        )

    def _hessian(self, sample: Array) -> Array:
        return self._loglike.hessian(sample) + self._prior.log_pdf_hessian(
            sample
        )
