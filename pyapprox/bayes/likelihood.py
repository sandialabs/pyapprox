"""
Likelihood Classes for Statistical Modeling

This module contains implementations of log-likelihood functions for various
probability distributions, including Bernoulli, Multinomial, and Exponential
distributions. These classes are designed to compute log-likelihoods,
generate random samples, and provide tools for derivative-based
optimization (e.g., Jacobians).

Purpose
-------
These classes are designed for use in statistical modeling, machine learning,
 and Bayesian inference.
They provide efficient computation of log-likelihoods and support
derivative-based methods for optimization and inference.

Features
--------
- Log-likelihood computation for various distributions.
- Random sample generation based on specified parameters.
- Support for Jacobian computation for optimization tasks.
- Integration with computational backends for numerical operations
  (e.g., Numpy, TensorFlow).

Usage
-----
These classes can be used in applications such as:
- Maximum likelihood estimation (MLE).
- Bayesian inference for parameter estimation.
- Model evaluation and comparison using likelihood-based metrics.
"""

from abc import abstractmethod

import numpy as np
from scipy import stats

from pyapprox.util.linalg import (
    inverse_of_cholesky_factor,
    log_determinant_from_cholesky_factor,
    diag_of_mat_mat_product,
)
from pyapprox.interface.model import Model, ChangeModelSignWrapper
from pyapprox.variables.joint import JointVariable
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.optimization.minimize import Optimizer
from pyapprox.optimization.scipy import ScipyConstrainedOptimizer


class LogLikelihood(Model):
    """
    A base class for computing the log-likelihood of a model given
    observations.

    This class is designed to be extended by specific implementations of
    log-likelihood computations for different models.
    While likelihoods are just probability distributions, using them to
    solve inverse problems, requires evaluating the log PDF
    distribution, e.g. :math:`p(x|s)` for many different realization of
    the shape parameters :math:`s`  of the distribution,
    e.g. the mean and covariance are shape parameters of a
    Gaussian distribution. The term shapes is also used loosely and may
    also include parameters of a distribution often called scales by scipy.
    This is in contrast the PDF of PyApprox's random variables which
    are evaluated at the independent variable of the PDF :math:`x`.
    Consequently, loglikeihood functions are written separately to allow
    computations to be fast for multiple shape parameters, where as
    variables are designed to be fast when being evaluated for the
    independent variable.
    Moreover, some likelihoods fix certain shape parameters and only
    allows sampling of a subset of the shape parameters, e.g.
    Gaussian Likelihoods used to estiamte an unknown mean for a fixed
    covariance

    Attributes
    ----------
    _obs : Array
        Stores the observations provided to the model.
    _nobs : int
        Number of observations.
    backend : BackendMixin
        Computational backend used for numerical operations
        (e.g., Numpy, TensorFlow).
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initializes the LogLikelihood class.

        Parameters
        ----------
        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to NumpyMixin.
        """
        super().__init__(backend=backend)

    def _check_obs_and_shapes(self, obs: Array, shapes: Array):
        if obs.shape[0] != shapes.shape[0]:
            raise ValueError(
                "obs and shapes must have the same number of rows (nobs)"
            )

    def _stack_obs_and_shapes(self, obs: Array, shapes: Array) -> Array:
        """
        Stacks repeated copies observations and shapes before log-likelihood
        computation.

        Parameters
        ----------
        obs : Array (nobs, nexperiments)
            Observations provided to the model.
        shapes : Array (nobs, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs: Array (nobs, nexperiments, nsamples)
            Processed observations.

        shapes: Array (nobs, nexperiments, nsamples)
            Processed shapes.

        Raises
        ------
        RuntimeError
            If `_setup()` has not been called.
        ValueError
            If the shapes' dimensions do not match the observations'
            dimensions.
        """
        self._check_obs_and_shapes(obs, shapes)
        nsamples = shapes.shape[1]
        nexperiments = obs.shape[1]
        obs = self._bkd.stack([obs] * nsamples, axis=-1)
        shapes = self._bkd.stack([shapes] * nexperiments, axis=1)
        if shapes.shape != obs.shape:
            msg = "shapes shape {0} does not match obs shape {1}".format(
                shapes.shape, obs.shape
            )
            raise ValueError(msg)
        return obs, shapes

    def _reshape_obs_and_shapes(self, obs: Array, shapes: Array) -> Array:
        """
        Reshapes observations and shapes before log-likelihood
        computations that use broadcasting.

        Parameters
        ----------
        obs : Array (nobs, nexperiments)
            Observations provided to the model.
        shapes : Array (nobs, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs: Array (nobs, nexperiments, nsamples)
            Processed observations.

        shapes: Array (nobs, nexperiments, nsamples)
            Processed shapes.

        Raises
        ------
        RuntimeError
            If `_setup()` has not been called.
        ValueError
            If the shapes' dimensions do not match the observations'
            dimensions.
        """
        self._check_obs_and_shapes(obs, shapes)
        return obs[..., None], shapes[:, None, :]

    def nvars(self) -> int:
        """
        Returns the number of unknown shape parameters

        Returns
        -------
        nobs: int
            Number of shape parameters
        """
        return self.nobs()

    @abstractmethod
    def nobs(self) -> int:
        """
        Returns the number of observations of a single experiment.
        For example if estimating the mean of a scalar Gaussian then nobs=1
        but if estimating a vector-valued mean then nobs is equal to the
        size of that vector.
        nobs is NOT the number of repititions (experiments).

        Returns
        -------
        nobs: int
            Number of observations.
        """
        raise NotImplementedError

    def nqoi(self) -> int:
        return 1

    def set_observations(self, obs: Array):
        """
        Sets the observations for the model.

        Parameters
        ----------
        obs : Array
            Observations provided to the model. Must be a 2D array with
            shape (nobs, nexperiments).

        Raises
        ------
        ValueError
            If the observations do not have the expected shape.
        """
        self._nobs, self._nexperiments = obs.shape
        if obs.ndim != 2 or obs.shape[0] != self.nobs():
            raise ValueError(
                "obs has the wrong shape {0} should be {1}".format(
                    obs.shape, (self.nobs(), 1)
                )
            )
        self._obs = obs

    def _check_shapes(self, shapes: Array):
        """
        Validates the dimensions of the shapes array.

        Parameters
        ----------
        shapes : Array (nobs, nsamples)
            Muliple realizations of the shape parameters of the likelihood
            distribution. If nexperiments > 1, then self._loglike_from_shapes() must
            use broadcasting (or tiling) to make the shapes of
            self._obs and shapes consistent. E.g. if obs shape is
            (nobs, nexperiments) and shapes.shape is (nobs, nsamples).
            then need to repeat obs nsamples times in third access so
            that obs.shape (nobs, nexperiments, nshapes) and shapes must be
            repeated nexperiment times, such that it also has that shape.
            This needs to be customized for each class because the best
            approach of broadcasting or repeating will depend on the liklihood.
            But self._stack_obs_and_shapes provides a default implementation.


        Raises
        ------
        ValueError
            If the shapes' dimensions do not match the number of observations.
        """
        if shapes.shape[0] != self.nobs():
            raise ValueError(
                "shapes has the wrong shape {0}".format(shapes.shape)
                + " nobs is {0}".format(self.nobs())
            )

    def loglike_from_shapes(self, shapes: Array) -> Array:
        """
        Computes the log-likelihood from muliple realizations of the shape
        parameters of the likelihood distribution

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Muliple realizations of the shape parameters of the likelihood
            distribution


        Returns
        -------
        values : Array (nsamples, 1)
            Log-likelihood values.
        """
        self._check_shapes(shapes)
        return self._loglike_from_shapes(shapes)

    def _values(self, shapes: Array) -> Array:
        return self.loglike_from_shapes(shapes)

    @abstractmethod
    def _rvs(self, shapes: Array) -> Array:
        """
        Abstract method for generating observations based on the
        likelihood distribution's shapes.

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def rvs(self, shapes: Array) -> Array:
        """
        Generate observations based on the
        likelihood distribution's shapes.

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        if shapes.ndim != 2 or shapes.shape[0] != self.nobs():
            raise ValueError(
                "shapes has the wrong shape. "
                f"Shape was {shapes.shape} but should should be 2D array with "
                f"{self.nobs()} rows"
            )
        nsamples = shapes.shape[1]
        obs = self._rvs(shapes)
        if obs.shape != (self.nobs(), nsamples):
            raise RuntimeError(
                "self._rvs returned obs with the wrong shape. "
                "Shape was {0} but should be {1}".format(
                    obs.shape, (nsamples, self.nobs())
                )
            )
        return obs


class ModelBasedLogLikelihoodMixin:
    """
    A mixin class for computing log-likelihoods based on a model that maps
    parameters to observations.

    This class provides methods for interacting with the model,
    computing Jacobians, Hessians, with respect to the shape parameters,
    and applying these derivatives to vectors
    """

    def model(self) -> Model:
        """
        Return the model used to map the parameters to the observations

        Returns
        -------
        model : Model
            The model used to map the parameters to the observations
        """
        return self._model

    def nvars(self) -> int:
        """
        Returns the number of model parameters which parameterize
        the unknown shapes of the likelihood (which are the qoi of the model)

        Returns
        -------
        nobs: int
            Number of shape parameters
        """
        return self._model.nvars()

    def nobs(self) -> int:
        """
        Returns the number of observations of a single experiment.
        For example if estimating the mean of a scalar Gaussian then nobs=1
        but if estimating a vector-valued mean then nobs is equal to the
        size of that vector.
        nobs is NOT the number of repititions (experiments).

        Returns
        -------
        nobs: int
            Number of observations.
        """

        return self._model.nqoi()

    def __repr__(self) -> str:
        return "{0}(nobs={1}, model={2})".format(
            self.__class__.__name__, self.nobs(), self._model
        )

    def _rvs(self, model_parameter_samples: Array) -> Array:
        return super()._rvs(self._model(model_parameter_samples).T)

    def _values(self, model_parameter_samples: Array):
        return self.loglike_from_shapes(self._model(model_parameter_samples).T)

    def rvs_from_shapes(self, shapes: Array) -> Array:
        """
        Generate observations based on the
        likelihood distribution's shapes,
        i.e. predictions of models at some samples.

        This function is useful for OED

        Parameters
        ----------
        shapes : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        return super()._rvs(shapes)

    def rvs(self, model_parameter_samples: Array) -> Array:
        """
        Generate observations based on the
        model parameter samples that determine the likelihood distribution's
        shapes.

        Parameters
        ----------
        model_parameter_samples : Array (nvars, nsamples)
            Predicted shapes or values from the model.

        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        if (
            model_parameter_samples.ndim != 2
            or model_parameter_samples.shape[0] != self._model.nvars()
        ):
            raise ValueError(
                "shapes has the wrong shape. "
                f"Shape was {model_parameter_samples.shape} "
                "but should should be 2D array with "
                f"{self.nobs()} rows"
            )
        nsamples = model_parameter_samples.shape[1]
        obs = self._rvs(model_parameter_samples)
        if obs.shape != (self.nobs(), nsamples):
            raise RuntimeError(
                "self._rvs returned obs with the wrong shape. "
                "Shape was {0} but should be {1}".format(
                    obs.shape, (nsamples, self.nobs())
                )
            )
        return obs


class GaussianLogLikelihood(LogLikelihood):
    """
    GaussianLikelihood that requires the application of the square-root
    of the known noise covariance matrix to vectors
    """

    def __init__(self, noise_cov: Array, backend: BackendMixin = NumpyMixin):
        """
        Initialize the GaussianLogLikelihood class.

        Parameters
        ----------
        noise_cov : Array (nobs, nobs)
            Dense noise covariance matrix.

        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to NumpyMixin.
        """
        super().__init__(backend=backend)
        self._setup(noise_cov)

    def _stack_obs_and_shapes(self, obs: Array, shapes: Array) -> Array:
        if not hasattr(self, "_loglike_const"):
            raise RuntimeError("must call _setup()")
        return super()._stack_obs_and_shapes(obs, shapes)

    def set_design_weights(self, design_weights: Array) -> Array:
        """
        Set the design weights that change the noise covariance to change
        the impact of each observations on the likelihood value.

        Parameters
        ----------
        design_weights : Array (nobs,)
            Design weights used to scale the noise covariance matrix.
        """
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
        self._loglike_const = -0.5 * (
            self._nobs * np.log(2.0 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov: Array):
        """
        Set up the noise covariance matrix and related properties.

        Parameters
        ----------
        noise_cov : Array (nobs, nobs)
            Dense noise covariance matrix.

        Raises
        ------
        ValueError
            If the noise covariance matrix is not square.
        """
        if noise_cov.ndim != 2 or noise_cov.shape[0] != noise_cov.shape[1]:
            raise ValueError(
                "noise_cov has the wrong shape {0}".format(noise_cov.shape)
            )
        self._noise_cov = noise_cov
        self._nobs = noise_cov.shape[0]
        self._last_L_inv_res = None
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def nobs(self) -> int:
        return self._nobs

    def _make_noisy(self, shapes: Array, noise: Array) -> Array:
        """
        Add noise to noiseless samples of the unknown mean (shapes).

        Parameters
        ----------
        shapes : Array (nobs, nsamples)
            Noiseless shapes.
        noise : Array (nobs, nsamples)
            Noise to be added.

        Returns
        -------
        noisy_shapes : Array (nobs, nsamples)
            Noisy shapes.

        Raises
        ------
        ValueError
            If the shapes of noiseless shapes and noise do not match.
        """
        if shapes.shape != noise.shape:
            msg = "shapes of shapes {0} and noise {1} must match".format(
                shapes.shape, noise.shape
            )
            raise ValueError(msg)
        return shapes + noise

    def _noise_cov_sqrt_apply(self, vecs: Array) -> Array:
        """
        Apply the square root of the noise covariance matrix to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise covariance square root is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol @ vecs

    def _noise_cov_sqrt_apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the tranpose of the square root of the noise covariance matrix
        to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the tranpose.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol.T @ vecs

    def _noise_cov_sqrt_inv_apply(self, vecs: Array) -> Array:
        """
        Apply the inverse of the square root of the noise covariance matrix
        to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise inverse is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol_inv @ vecs

    def _noise_cov_sqrt_inv_apply_transpose(self, vecs: Array) -> Array:
        """
        Apply the transpose of the inverse of the square root of the
        noise covariance matrix to vectors.

        Parameters
        ----------
        vecs : Array (nobs, nvecs)
            Vectors to which the noise inverse transpose is applied.

        Returns
        -------
        values : Array (nobs, nvecs)
            Result of the application.
        """
        return self._wnoise_chol_inv.T @ vecs

    def _sample_noise(self, nsamples: int) -> Array:
        """
        Sample noise from the Gaussian distribution.

        Parameters
        ----------
        nsamples : int
            Number of noise samples to generate.

        Returns
        -------
        noise : Array (nobs, nsamples)
            Noise samples.
        """
        # create samples (nsamples, nobs) then take transpose
        # to ensure same noise is used for ith sample
        # regardless of size of nsam
        normal_samples = self._bkd.asarray(
            np.random.normal(0, 1, (nsamples, self.nobs()))
        ).T
        return self._noise_cov_sqrt_apply(normal_samples)

    def _check_residuals(self, residuals: Array):
        if residuals.ndim != 3:
            raise ValueError(
                "residuals must be a 3D array with shape "
                "(nobs, nexperiments, nsamples)"
            )

    def _loglike_from_residuals(self, residuals: Array) -> Array:
        """
        Compute the log-likelihood from residuals, i.e. the difference
        between the observations and the shapes

        Parameters
        ----------
        many_obs : Array
            Multiple observations.
        shapes : Array
            Predicted shapes or values.

        Returns
        -------
        Array
            Log-likelihood values for each observation.
        """
        self._check_residuals(residuals)
        nexperiments = residuals.shape[1]
        vals = 0.0
        for ii in range(nexperiments):
            residual = residuals[:, ii]
            L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
            vals += diag_of_mat_mat_product(
                L_inv_res.T, L_inv_res, bkd=self._bkd
            )
        return (-0.5 * vals + self._loglike_const * nexperiments)[:, None]

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        if self._obs.shape != (self.nobs(), self._nexperiments):
            raise ValueError(
                "self._obs shape was {0} but should be {1}".format(
                    self._obs.shape, (self.nobs(), self._nexperiments)
                )
            )
        if shapes.ndim != 2:
            raise ValueError("shapes must be a 2D array")
        # obs, shapes = self._stack_obs_and_shapes(self._obs, shapes)
        obs, shapes = self._reshape_obs_and_shapes(self._obs, shapes)
        return self._loglike_from_residuals(obs - shapes)

    def _jacobian(self, model_parameter_sample: Array) -> Array:
        shapes = self._model(model_parameter_sample).T
        residual = self._obs - shapes
        L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
        return self._noise_cov_sqrt_inv_apply(L_inv_res).T

    def _apply_jacobian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        shapes = self._model(model_parameter_sample).T
        residual = self._obs - shapes
        L_inv_res = self._noise_cov_sqrt_inv_apply(residual)
        return L_inv_res.T @ self._noise_cov_sqrt_inv_apply(vec)

    def _apply_hessian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        return self._bkd.zeros(vec.shape)

    def noise_covariance(self) -> Array:
        """
        Return the weighted noise covariance matrix.

        Returns
        -------
        noise_cov : Array (nobs, nobs)
            Weighted noise covariance matrix.
        """
        return self._wnoise_chol @ self._wnoise_chol.T

    def _rvs_from_likelihood_samples(
        self, shapes: Array, latent_samples: Array
    ) -> Array:
        """
        Generate observations based on the
        likelihood distribution's shapes, ,
        i.e. predictions of models at some samples,
        and samples from the latent space
        of the likelihood. For general independent variables the latent
        space is the uniform measure on [0,1]^nvars. These samples are then
        mapped to the likelihood using the inverse CDF of the likelihood.
        However, for certain distribution, simpler transformations can be used
        using different latent variables, For easmple, the latent variable of
        the Gaussian likelihood is samples of the noise variable which
        are added to the mean (shapes) of the likelihood.

        This function is useful for OED

        Parameters
        ----------
        shapes : Array (nobs, nsamples)
            Predicted shapes or values from the model.

        latent_samples : Array (nobs, nsamples)
            Predicted shapes or values from the model.


        Returns
        -------
        obs : Array (nsamples, nobs)
            Generated observations.
        """
        if latent_samples.shape != shapes.shape:
            raise ValueError(
                f"shapes of latent_samples {latent_samples.shape} "
                f"and shapes {shapes.shape} do not match"
                ""
            )
        return self._make_noisy(shapes, latent_samples)

    def _rvs(self, shapes: Array) -> Array:
        return self._rvs_from_likelihood_samples(
            shapes, self._sample_noise(shapes.shape[1])
        )


class IndependentGaussianLogLikelihood(GaussianLogLikelihood):
    """
    A likelihood function associated with adding independent
    but not necessarily identically distributed Gaussian noise.
    Evaluating this is faster than GaussianLikelihood if noise covariance
    is a diagonal matrix
    """

    def __init__(
        self, noise_cov_diag: Array, backend: BackendMixin = NumpyMixin
    ):
        """
        Initialize the IndependentGaussianLogLikelihood class.

        Parameters
        ----------
        noise_cov_diag : Array (nobs,)
            Diagonal noise covariance matrix.

        backend : BackendMixin, optional
            Computational backend for numerical operations.
            Defaults to NumpyMixin.
        """
        super().__init__(noise_cov_diag, backend=backend)

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
        self._loglike_const = -0.5 * (
            self._nobs * np.log(2.0 * np.pi) + self._log_wnoise_cov_det
        )

    def _setup(self, noise_cov_diag: Array):
        if noise_cov_diag.ndim != 2 or noise_cov_diag.shape[1] != 1:
            raise ValueError(
                "noise_cov_diag has the wrong shape {0}".format(
                    noise_cov_diag.shape
                )
            )
        self._nobs = noise_cov_diag.shape[0]
        self._noise_cov_diag = noise_cov_diag
        self.set_design_weights(self._bkd.ones((self._nobs, 1)))

    def _noise_cov_sqrt_apply(self, vecs: Array) -> Array:
        return self._wnoise_std_diag * vecs

    def _noise_cov_sqrt_apply_transpose(self, vecs: Array) -> Array:
        return self._noise_cov_sqrt_apply(vecs)

    def _noise_cov_sqrt_inv_apply(self, vecs: Array) -> Array:
        return self._bkd.sqrt(self._wnoise_cov_inv_diag) * vecs

    def _noise_cov_sqrt_inv_apply_transpose(self, vecs: Array) -> Array:
        return self._noise_cov_sqrt_inv_apply(vecs)

    def noise_covariance(self) -> Array:
        return self._bkd.diag(1.0 / self._wnoise_cov_inv_diag[:, 0])

    def __repr__(self) -> str:
        return "{0}(nobs={1}, sigma={2})".format(
            self.__class__.__name__,
            self.nobs(),
            self._noise_std_diag[:, 0],
        )


class ModelBasedGaussianLogLikelihoodMixin(ModelBasedLogLikelihoodMixin):
    def jacobian_implemented(self) -> bool:
        return self._model.jacobian_implemented()

    def apply_jacobian_implemented(self) -> bool:
        return self._model.apply_jacobian_implemented()

    def apply_hessian_implemented(self) -> bool:
        return self._model.hessian_implemented()

    def _jacobian(self, model_parameter_sample: Array) -> Array:
        shapes = self._model(model_parameter_sample).T
        obs, shapes = self._reshape_obs_and_shapes(self._obs, shapes)
        residual = obs - shapes
        nexperiments = obs.shape[1]
        jac = 0.0
        for ii in range(nexperiments):
            L_inv_res = self._noise_cov_sqrt_inv_apply(residual[:, ii])
            jac += L_inv_res.T @ self._noise_cov_sqrt_inv_apply(
                self._model.jacobian(model_parameter_sample)
            )
        return jac

    def _apply_jacobian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        shapes = self._model(model_parameter_sample).T
        obs, shapes = self._reshape_obs_and_shapes(self._obs, shapes)
        residual = obs - shapes
        nexperiments = obs.shape[1]
        jvp = 0.0
        for ii in range(nexperiments):
            L_inv_res = self._noise_cov_sqrt_inv_apply(residual[:, ii])
            jvp += L_inv_res.T @ self._noise_cov_sqrt_inv_apply(
                self._model.apply_jacobian(model_parameter_sample, vec)
            )
        return jvp

    def _apply_hessian(
        self, model_parameter_sample: Array, vec: Array
    ) -> Array:
        shapes = self._model(model_parameter_sample).T
        # use adjoint method to compute hvp with
        # objective f(s,z) = -1/2 s.T @ s
        obs, shapes = self._reshape_obs_and_shapes(self._obs, shapes)
        residual = obs - shapes
        nexperiments = obs.shape[1]
        hvp = 0.0
        for ii in range(nexperiments):
            # solve forward equation for state s
            # c(s,z) = Gs - obs + model(z) = 0, G=wnoise_chol_inv
            state = self._noise_cov_sqrt_inv_apply(residual[:, ii])
            # solve adjoint solution
            # lambda = -inv(c_s.T)f_s = -inv(G.T)s
            lamda = self._noise_cov_sqrt_inv_apply_transpose(state)
            tmp1 = self._noise_cov_sqrt_inv_apply(
                self._model.jacobian(model_parameter_sample)
            )
            # Hvp = c_z.T @ p + L_zs @ w + L_zz @ v
            # w = inv(c_s)@ c_z @ v = inv(G) @ model.jacobian(z) @ v
            # L_ss = -I, L_zs = 0, L_sz = 0, L_zz = lamda.T @ model.hessian(z)
            # p = -inv(G.T) @ w
            hvp += -tmp1.T @ (tmp1 @ vec) + self._model.apply_weighted_hessian(
                model_parameter_sample, vec, lamda
            )
        return hvp


class ModelBasedGaussianLogLikelihood(
    ModelBasedGaussianLogLikelihoodMixin, GaussianLogLikelihood
):
    """
    A likelihood function for adding correlated Gaussian noise to observations,
    using a model that maps parameters to observations.
    """

    def __init__(self, model: Model, noise_cov: Array):
        """
        Initialize the ModelBasedGaussianLogLikelihood class.

        Parameters
        ----------
        model : Model
            The model used to map parameters to observations.

        noise_cov : Array (nobs, nobs)
            Dense noise covariance matrix.
        """
        super().__init__(noise_cov, backend=model._bkd)
        self._model = model


class ModelBasedIndependentGaussianLogLikelihood(
    ModelBasedGaussianLogLikelihoodMixin, IndependentGaussianLogLikelihood
):
    """
    A likelihood function for adding independent (but not necessarily
    identically distributed) Gaussian noise to observations, using a model
    that maps parameters to observations.

    Evaluating this is faster than GaussianLikelihood if noise covariance
    is a diagonal matrix
    """

    def __init__(self, model: Model, noise_cov_diag: Array):
        """
        Initialize the ModelBasedIndependentGaussianLogLikelihood class.

        Parameters
        ----------
        model : Model
            The model used to map parameters to observations.

        noise_cov_diag : Array (nobs,)
            Diagonal noise covariance matrix.
        """
        super().__init__(noise_cov_diag, backend=model._bkd)
        self._model = model


class IndependentExponentialLogLikelihood(LogLikelihood):
    r"""
    Log-likelihood for exponentially distributed variables with an unknown
    rate parameter where observations of the time to each event is assumed
    independent

    This class models observations as times between consecutive events,
    where the eventsfollow an exponential distribution.
    The rate parameter determines the rate at which events occur over time.
    """

    def nobs(self) -> int:
        """
        Returns the number of observations.

        Notes
        -----
        - `nobs` can be considered as 1, with multiple experiments taken.
        - Alternatively, `nobs` can be the number of observations from a
          single experiment.
        - This implementation uses the former convention.

        Returns
        -------
        int
            Number of observations (always 1 in this implementation).
        """
        return 1
        return 1

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        """
        Computes the log-likelihood for the given rate parameters.

        Parameters
        ----------
        shapes : Array
            Rate parameters for the exponential distribution.

        Returns
        -------
        Array
            Log-likelihood values for the given rate parameters.

        Raises
        ------
        RuntimeError
            If observations (`_obs`) have not been set using
            `set_observations()`.
        """
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        # nobs can be considered 1 and n multiple experiments are taken
        # or nobs=n using only a single experiments is taken
        # we choose the later convention
        return (
            self._bkd.log(shapes).sum(axis=0)
            - (self._obs * shapes).sum(axis=0)
        )[:, None]

    def _make_noisy(self, shapes: Array, noise: Array) -> Array:
        """
        Generates noisy observations based on the rate parameters and noise.

        Parameters
        ----------
        shapes : Array
            Rate parameters for the exponential distribution.
        noise : Array
            Noise sampled from an exponential distribution with unit mean.

        Returns
        -------
        Array
            Noisy observations.

        Raises
        ------
        ValueError
            If the shapes of `shapes` and `noise` do not match.

        Notes
        -----
        - Observations are generated as `noise / shapes`.
        - Given exponentially distributed noise with unit mean, this method
          returns realizations exponentially distributed with means specified
          by `shapes`.
        """
        if shapes.shape != noise.shape:
            msg = f"sizes of shapes {shapes.shape}"
            f"and noise {noise.shape} must match"
            raise ValueError(msg)
        return noise / shapes

    def _sample_noise(self, nsamples: int) -> Array:
        """
        Samples noise from an exponential distribution with unit mean.

        Parameters
        ----------
        nsamples : int
            Number of noise samples to generate.

        Returns
        -------
        Array
            Noise samples.

        Notes
        -----
        - Noise is sampled from an exponential distribution with mean 1.
        - The samples are transposed to ensure the same noise is used for the
          `i`-th sample, regardless of the size of `nsamples`.
        """
        exponential_samples = self._bkd.asarray(
            np.random.exponential(1, (nsamples, self.nobs()))
        ).T
        return exponential_samples

    def _rvs(self, shapes: Array) -> Array:
        """
        Generates random samples from the exponential distribution based on
        the rate parameters.

        Parameters
        ----------
        shapes : Array
            Rate parameters for the exponential distribution.

        Returns
        -------
        Array
            Random samples from the exponential distribution.

        Notes
        -----
        - Random samples are generated as `noise / shapes`, where `noise` is
          sampled from an exponential distribution with unit mean.
        """
        return self._make_noisy(shapes, self._sample_noise(shapes.shape[1]))


class BernoulliLogLikelihood(LogLikelihood):
    """
    This class computes the log-likelihood for observations modeled as
    Bernoulli random variables, where the probability of success (1) is
    given by the parameter `shapes`. The Bernoulli distribution
    is commonly used to model binary outcomes.
    """

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        # single vector realization of obs forms one column
        # different realizations in different colums
        log_vals = self._bkd.sum(
            self._bkd.log(shapes) * self._obs.T
            + self._bkd.log(1.0 - shapes) * (1.0 - self._obs.T),
            axis=0,
        )
        return log_vals

    def nobs(self) -> int:
        return 1

    def _rvs(self, shapes: Array) -> Array:
        """
        Generates random samples from the Bernoulli distribution based on the
        given probabilities.

        Parameters
        ----------
        shapes : Array
            Probabilities of success (1) for the Bernoulli distribution.

        Returns
        -------
        Array
            Random samples (binary outcomes: 0 or 1) from the Bernoulli
            distribution.

        Notes
        -----
        - Random samples are generated using the probabilities specified in
          `shapes`.
        - Each column in `shapes` corresponds to a separate realization.
        """
        obs = self._bkd.stack(
            [
                self._bkd.asarray(stats.bernoulli(probs).rvs(1)[0])
                for probs in shapes.T
            ],
            axis=0,
        )
        return obs

    def _values(self, samples: Array) -> Array:
        return self._loglike_from_shapes(samples)[:, None]

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[:, 0], sample[:, 0]
        )


class MultinomialLogLikelihood(LogLikelihood):
    """
    Log-likelihood for multinomial-distributed observations.

    This class computes the log-likelihood for observations modeled as
    multinomial random variables, where the probabilities of each category are
    given by the parameter `shapes`. The multinomial
    distribution is commonly used to model categorical outcomes over multiple
    trials.
    """

    def __init__(
        self, noptions: int, ntrials: int, backend: BackendMixin = NumpyMixin
    ):
        """
        Initialize the MultinomialLogLikelihood class.

        Parameters
        ----------
        noptions : int
            Number of categories (options) in the multinomial distribution.
        ntrials : int
            Number of trials for the multinomial distribution.
        backend : BackendMixin, optional
            Computational backend for numerical operations. Defaults to `NumpyMixin`.
        """
        super().__init__(backend)
        self._noptions = noptions
        self._ntrials = self._bkd.asarray(ntrials)

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        if self._obs is None:
            raise RuntimeError("must call set_observations()")
        # single vector realization of obs forms one column
        # different realizations in different colums
        if self._obs.shape[0] != self.nobs():
            raise RuntimeError("Obs has the wrong shape")
        # log(factorial(N) = gammaln(N+1)
        nexperiments = self._obs.shape[1]
        const = nexperiments * self._bkd.gammaln(self._ntrials + 1)
        log_vals = self._bkd.log(shapes).T @ self._obs - self._bkd.sum(
            self._bkd.gammaln(self._obs + 1), axis=0
        )
        # take product (sum in log space) over all experiments
        return self._bkd.sum(log_vals, axis=1) + const

    def nobs(self) -> int:
        """
        Returns the number of categories (options).

        Returns
        -------
        nobs : int
            Number of categories (options) in the multinomial distribution.
        """
        return self._noptions

    def _rvs(self, shapes: Array) -> Array:
        obs = self._bkd.stack(
            [
                self._bkd.asarray(
                    stats.multinomial(self._ntrials, probs).rvs(1)[0]
                )
                for probs in shapes.T
            ],
            axis=0,
        )
        return obs

    def _values(self, samples: Array) -> Array:
        return self._loglike_from_shapes(samples)[:, None]

    def __repr__(self) -> str:
        return "{0}(noptions={1})".format(
            self.__class__.__name__, self.nvars()
        )

    def jacobian_implemented(self) -> bool:
        return self._bkd.jacobian_implemented()

    def _jacobian(self, sample: Array) -> Array:
        return self._bkd.jacobian(
            lambda x: self._values(x[:, None])[:, 0], sample[:, 0]
        )


class LogUnNormalizedPosterior(Model):
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
        return (
            self._prior.pdf_hessian_implemented()
            and self._loglike.hessian_implemented()
        )

    def apply_hessian_implemented(self) -> bool:
        return (
            self._prior.pdf_hessian_implemented()
            and self._loglike.apply_hessian_implemented()
        )

    def _values(self, samples: Array) -> Array:
        return self._loglike(samples) + self._prior.logpdf(samples)

    def _jacobian(self, sample: Array) -> Array:
        return self._loglike.jacobian(sample) + self._prior.logpdf_jacobian(
            sample
        )

    def _hessian(self, sample: Array) -> Array:
        return self._loglike.hessian(sample) + self._prior.logpdf_hessian(
            sample
        )

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        return (
            self._loglike.apply_hessian(sample, vec)
            + self._prior.logpdf_hessian(sample) @ vec
        )

    def set_optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer
        self._optimizer.set_objective_function(ChangeModelSignWrapper(self))

    def default_optimizer(self) -> Optimizer:
        local_optimizer = ScipyConstrainedOptimizer()
        # global_optimizer = ScipyConstrainedDifferentialEvolutionOptimizer(
        #     opts={"maxiter": 100}
        # )
        # # only set bounds for differential evolution because it is required
        # # by algorithm
        # global_optimizer.set_bounds(self._prior.truncated_ranges(1 - 1e-3))
        # optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        return local_optimizer

    def maximum_aposteriori_point(self, iterate: Array = None) -> Array:
        if iterate is None:
            iterate = self._prior.mean()
        if not hasattr(self, "_optimizer"):
            self.set_optimizer(self.default_optimizer())
        res = self._optimizer.minimize(iterate)
        return res.x


class ExponentialQuarticLogLikelihoodModel(LogLikelihood):
    # likelihood function sometimes found in the literature
    # not a true likelihood so only implements a subset of
    # functions
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend=backend)
        self._a = 3.0

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 1

    def _values(self, x: Array) -> Array:
        value = -(0.1 * x[0] ** 4 + 0.5 * (2.0 * x[1] - x[0] ** 2) ** 2)
        return value[:, None]

    def _jacobian(self, x: Array):
        grad = -self._bkd.array(
            [
                12.0 / 5.0 * x[0] ** 3 - 4.0 * x[0] * x[1],
                4.0 * x[1] - 2.0 * x[0] ** 2,
            ]
        )
        return grad

    def jacobian_implemented(self) -> bool:
        return True

    def _loglike_from_shapes(self, shapes: Array) -> Array:
        raise NotImplementedError

    def _rvs(self):
        raise NotImplementedError

    def nobs(self) -> int:
        return 2
