"""
Gaussian likelihood functions.

Provides Gaussian log-likelihood functions for Bayesian inference.
"""

from typing import Generic, Optional
import math

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols.covariance import (
    SqrtCovarianceOperatorProtocol,
)
from pyapprox.probability.covariance import DiagonalCovarianceOperator


class GaussianLogLikelihood(Generic[Array]):
    """
    Gaussian log-likelihood with noise covariance operator.

    For Gaussian noise model:
        obs = model_output + noise, noise ~ N(0, noise_cov)

    Log-likelihood:
        log p(obs | model) = -0.5 * ||L^{-1}(obs - model)||^2
                             - 0.5 * log|noise_cov| - nobs/2 * log(2*pi)

    where noise_cov = L @ L.T.

    Parameters
    ----------
    noise_cov_op : SqrtCovarianceOperatorProtocol[Array]
        Noise covariance operator.
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.covariance import (
    ...     DenseCholeskyCovarianceOperator
    ... )
    >>> bkd = NumpyBkd()
    >>> noise_cov = 0.01 * np.eye(3)
    >>> noise_op = DenseCholeskyCovarianceOperator(noise_cov, bkd)
    >>> likelihood = GaussianLogLikelihood(noise_op, bkd)
    >>> likelihood.set_observations(np.array([[1.0], [2.0], [3.0]]))
    >>> model_outputs = np.array([[1.01], [1.99], [3.02]])
    >>> logpdf = likelihood.logpdf(model_outputs)
    """

    def __init__(
        self,
        noise_cov_op: SqrtCovarianceOperatorProtocol[Array],
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._noise_cov_op = noise_cov_op
        self._nobs = noise_cov_op.nvars()
        self._observations: Optional[Array] = None
        self._design_weights: Optional[Array] = None

        # Pre-compute normalization constant
        self._log_norm_const = self._compute_log_normalization()

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nobs(self) -> int:
        """Return the number of observations."""
        return self._nobs

    def noise_covariance_operator(
        self,
    ) -> SqrtCovarianceOperatorProtocol[Array]:
        """Get the noise covariance operator."""
        return self._noise_cov_op

    def set_observations(self, obs: Array) -> None:
        """
        Set the observed data.

        Parameters
        ----------
        obs : Array
            Observed data. Shape: (nobs,) or (nobs, 1)
        """
        if obs.ndim == 1:
            obs = self._bkd.reshape(obs, (self._nobs, 1))
        if obs.shape[0] != self._nobs:
            raise ValueError(
                f"Observations have wrong shape: expected ({self._nobs}, ?), "
                f"got {obs.shape}"
            )
        self._observations = obs

    def set_design_weights(self, weights: Array) -> None:
        """
        Set weights for experimental design.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs,)
        """
        if weights.shape[0] != self._nobs:
            raise ValueError(
                f"Weights have wrong shape: expected ({self._nobs},), "
                f"got {weights.shape}"
            )
        self._design_weights = weights

    def _validate_model_outputs(self, model_outputs: Array) -> None:
        """Validate that model_outputs is 2D with correct shape."""
        if model_outputs.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nobs, nsamples), "
                f"got {model_outputs.ndim}D"
            )
        if model_outputs.shape[0] != self._nobs:
            raise ValueError(
                f"Expected {self._nobs} observations, got {model_outputs.shape[0]}"
            )

    def logpdf(self, model_outputs: Array) -> Array:
        """
        Evaluate the log-likelihood.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples) - must be 2D

        Returns
        -------
        Array
            Log-likelihood values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If observations not set or model_outputs has wrong shape
        """
        if self._observations is None:
            raise ValueError(
                "Observations not set. Call set_observations first."
            )

        self._validate_model_outputs(model_outputs)

        # Residuals: obs - model
        residuals = self._observations - model_outputs

        # Apply design weights if set
        if self._design_weights is not None:
            residuals = residuals * self._design_weights[:, None]

        # Whitened residuals: L^{-1} @ residuals
        whitened = self._noise_cov_op.apply_inv(residuals)

        # Squared Mahalanobis distance: ||L^{-1}(obs - model)||^2
        squared_dist = self._bkd.sum(whitened**2, axis=0)

        result = self._log_norm_const - 0.5 * squared_dist
        return self._bkd.reshape(result, (1, -1))

    def logpdf_vectorized(
        self, model_outputs: Array, observations: Array
    ) -> Array:
        """
        Batched log-likelihood evaluation.

        Computes log p(obs | model) for all combinations.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, n_model_samples) - must be 2D
        observations : Array
            Observed data. Shape: (nobs, n_obs_samples) - must be 2D

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (n_model_samples, n_obs_samples)

        Raises
        ------
        ValueError
            If inputs are not 2D
        """
        self._validate_model_outputs(model_outputs)
        if observations.ndim != 2:
            raise ValueError(
                f"Expected 2D observations array, got {observations.ndim}D"
            )

        n_model = model_outputs.shape[1]
        n_obs = observations.shape[1]

        # Compute residuals for all (model, obs) combinations via broadcasting
        # model_outputs: (nobs, n_model) -> (nobs, n_model, 1)
        # observations: (nobs, n_obs) -> (nobs, 1, n_obs)
        # residuals: (nobs, n_model, n_obs)
        residuals = (
            observations[:, None, :] - model_outputs[:, :, None]
        )

        if self._design_weights is not None:
            residuals = residuals * self._design_weights[:, None, None]

        # Reshape to 2D for apply_inv: (nobs, n_model * n_obs)
        residuals_2d = self._bkd.reshape(residuals, (self._nobs, n_model * n_obs))

        # Apply whitening transformation
        whitened_2d = self._noise_cov_op.apply_inv(residuals_2d)

        # Reshape back to 3D: (nobs, n_model, n_obs)
        whitened = self._bkd.reshape(whitened_2d, (self._nobs, n_model, n_obs))

        # Sum squared over nobs dimension -> (n_model, n_obs)
        squared_dist = self._bkd.sum(whitened**2, axis=0)

        return self._log_norm_const - 0.5 * squared_dist

    def rvs(self, model_outputs: Array, nsamples: int = 1) -> Array:
        """
        Sample from the likelihood (add noise to model outputs).

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, n_model_samples) - must be 2D
        nsamples : int
            Number of noise samples per model output.

        Returns
        -------
        Array
            Noisy observations. Shape: (nobs, n_model_samples * nsamples)

        Raises
        ------
        ValueError
            If model_outputs is not 2D
        """
        self._validate_model_outputs(model_outputs)

        n_model = model_outputs.shape[1]
        total_samples = n_model * nsamples

        # Generate standard normal samples
        std_normal = self._bkd.asarray(
            np.random.normal(0, 1, (self._nobs, total_samples))
        )

        # Apply covariance structure: noise = L @ z
        noise = self._noise_cov_op.apply(std_normal)

        # Add noise to model outputs (tile model outputs if needed)
        if nsamples > 1:
            model_tiled = self._bkd.tile(model_outputs, (1, nsamples))
        else:
            model_tiled = model_outputs

        return model_tiled + noise

    def _compute_log_normalization(self) -> float:
        """Compute log normalization constant."""
        # -nobs/2 * log(2*pi) - 0.5 * log|noise_cov|
        # = -nobs/2 * log(2*pi) - log|L| (since |Cov| = |L|^2)
        log_det_L = self._noise_cov_op.log_determinant()
        return float(-0.5 * self._nobs * math.log(2 * math.pi) - log_det_L)

    def gradient(self, model_outputs: Array) -> Array:
        """
        Compute gradient of log-likelihood w.r.t. model outputs.

        d/d(model) log p = Cov^{-1} @ (obs - model)

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples) - must be 2D

        Returns
        -------
        Array
            Gradient. Shape: (nobs, nsamples)

        Raises
        ------
        ValueError
            If observations not set or model_outputs has wrong shape
        """
        if self._observations is None:
            raise ValueError("Observations not set.")

        self._validate_model_outputs(model_outputs)

        residuals = self._observations - model_outputs

        if self._design_weights is not None:
            residuals = residuals * self._design_weights[:, None]

        # Gradient = Cov^{-1} @ residuals = L^{-T} @ L^{-1} @ residuals
        whitened = self._noise_cov_op.apply_inv(residuals)
        return self._noise_cov_op.apply_inv_transpose(whitened)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GaussianLogLikelihood(nobs={self._nobs})"


class DiagonalGaussianLogLikelihood(Generic[Array]):
    """
    Gaussian log-likelihood with diagonal (independent) noise.

    Specialized implementation for uncorrelated noise with
    possibly different variances per observation.

    Parameters
    ----------
    noise_variances : Array
        Noise variances for each observation. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> noise_var = np.array([0.01, 0.02, 0.01])
    >>> likelihood = DiagonalGaussianLogLikelihood(noise_var, bkd)
    >>> likelihood.set_observations(np.array([1.0, 2.0, 3.0]))
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._variances = noise_variances
        self._nobs = noise_variances.shape[0]
        self._observations: Optional[Array] = None
        self._design_weights: Optional[Array] = None

        # Pre-compute for efficiency
        self._std = bkd.sqrt(noise_variances)
        self._inv_var = 1.0 / noise_variances
        self._log_norm_const = self._compute_log_normalization()

        # Create diagonal covariance operator for protocol compliance
        self._noise_cov_op = DiagonalCovarianceOperator(noise_variances, bkd)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nobs(self) -> int:
        """Return the number of observations."""
        return self._nobs

    def noise_covariance_operator(
        self,
    ) -> SqrtCovarianceOperatorProtocol[Array]:
        """Get the noise covariance operator."""
        return self._noise_cov_op

    def set_observations(self, obs: Array) -> None:
        """Set the observed data."""
        if obs.ndim == 1:
            obs = self._bkd.reshape(obs, (self._nobs, 1))
        if obs.shape[0] != self._nobs:
            raise ValueError(
                f"Observations have wrong shape: expected ({self._nobs}, ?), "
                f"got {obs.shape}"
            )
        self._observations = obs

    def set_design_weights(self, weights: Array) -> None:
        """Set weights for experimental design."""
        if weights.shape[0] != self._nobs:
            raise ValueError(
                f"Weights have wrong shape: expected ({self._nobs},), "
                f"got {weights.shape}"
            )
        self._design_weights = weights

    def _validate_model_outputs(self, model_outputs: Array) -> None:
        """Validate that model_outputs is 2D with correct shape."""
        if model_outputs.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (nobs, nsamples), "
                f"got {model_outputs.ndim}D"
            )
        if model_outputs.shape[0] != self._nobs:
            raise ValueError(
                f"Expected {self._nobs} observations, got {model_outputs.shape[0]}"
            )

    def logpdf(self, model_outputs: Array) -> Array:
        """
        Evaluate the log-likelihood.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples) - must be 2D

        Returns
        -------
        Array
            Log-likelihood values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If observations not set or model_outputs has wrong shape
        """
        if self._observations is None:
            raise ValueError("Observations not set.")

        self._validate_model_outputs(model_outputs)

        residuals = self._observations - model_outputs

        if self._design_weights is not None:
            weighted_inv_var = self._inv_var * self._design_weights**2
            squared_dist = self._bkd.sum(
                residuals**2 * weighted_inv_var[:, None], axis=0
            )
        else:
            squared_dist = self._bkd.sum(
                residuals**2 * self._inv_var[:, None], axis=0
            )

        result = self._log_norm_const - 0.5 * squared_dist
        return self._bkd.reshape(result, (1, -1))

    def rvs(self, model_outputs: Array, nsamples: int = 1) -> Array:
        """
        Sample from the likelihood.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, n_model_samples) - must be 2D
        nsamples : int
            Number of noise samples per model output.

        Returns
        -------
        Array
            Noisy observations. Shape: (nobs, n_model_samples * nsamples)

        Raises
        ------
        ValueError
            If model_outputs is not 2D
        """
        self._validate_model_outputs(model_outputs)

        n_model = model_outputs.shape[1]
        total_samples = n_model * nsamples

        # Generate noise
        std_normal = self._bkd.asarray(
            np.random.normal(0, 1, (self._nobs, total_samples))
        )
        noise = self._std[:, None] * std_normal

        if nsamples > 1:
            model_tiled = self._bkd.tile(model_outputs, (1, nsamples))
        else:
            model_tiled = model_outputs

        return model_tiled + noise

    def _compute_log_normalization(self) -> float:
        """Compute log normalization constant."""
        log_det = float(self._bkd.sum(self._bkd.log(self._variances)))
        return float(-0.5 * self._nobs * math.log(2 * math.pi) - 0.5 * log_det)

    def gradient(self, model_outputs: Array) -> Array:
        """
        Compute gradient of log-likelihood w.r.t. model outputs.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, nsamples) - must be 2D

        Returns
        -------
        Array
            Gradient. Shape: (nobs, nsamples)

        Raises
        ------
        ValueError
            If observations not set or model_outputs has wrong shape
        """
        if self._observations is None:
            raise ValueError("Observations not set.")

        self._validate_model_outputs(model_outputs)

        residuals = self._observations - model_outputs
        return self._inv_var[:, None] * residuals

    def logpdf_vectorized(
        self, model_outputs: Array, observations: Array
    ) -> Array:
        """
        Batched log-likelihood evaluation.

        Computes log p(obs | model) for all combinations.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: (nobs, n_model_samples) - must be 2D
        observations : Array
            Observed data. Shape: (nobs, n_obs_samples) - must be 2D

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (n_model_samples, n_obs_samples)

        Raises
        ------
        ValueError
            If inputs are not 2D
        """
        self._validate_model_outputs(model_outputs)
        if observations.ndim != 2:
            raise ValueError(
                f"Expected 2D observations array, got {observations.ndim}D"
            )

        # Residuals for all (model, obs) combinations via broadcasting
        # model_outputs: (nobs, n_model, 1), observations: (nobs, 1, n_obs)
        # residuals: (nobs, n_model, n_obs)
        residuals = observations[:, None, :] - model_outputs[:, :, None]

        if self._design_weights is not None:
            weighted_inv_var = self._inv_var * self._design_weights**2
            sqrt_w = self._bkd.sqrt(weighted_inv_var)
        else:
            sqrt_w = self._bkd.sqrt(self._inv_var)

        scaled = sqrt_w[:, None, None] * residuals
        squared_dist = self._bkd.einsum("ijk,ijk->jk", scaled, scaled)

        return self._log_norm_const - 0.5 * squared_dist

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DiagonalGaussianLogLikelihood(nobs={self._nobs})"


class MultiExperimentLogLikelihood(Generic[Array]):
    """
    Sum of log-likelihoods across multiple experiments.

    Wraps a likelihood that supports ``logpdf_vectorized()`` and evaluates
    ``sum_j log p(obs_j | model)`` for each model sample in one vectorized
    call.

    This class assumes a **shared forward model** across all experiments:
    the same model predictions are evaluated against each observation set.
    It operates on model outputs (predictions), not latent variables.
    Callers map latent variables to predictions themselves, e.g.
    ``multi_lik.logpdf(obs_matrix @ z)``.

    For per-experiment experimental conditions (different forward models
    per experiment), use amortized VI with separate log-likelihood
    callables per group instead.

    Parameters
    ----------
    base_likelihood : VectorizedLogLikelihoodProtocol[Array]
        Likelihood with ``logpdf_vectorized(model_outputs, observations)``.
    observations : Array
        Multiple observation datasets. Shape: ``(nobs, nexperiments)``
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        base_likelihood: "VectorizedLogLikelihoodProtocol[Array]",
        observations: Array,
        bkd: Backend[Array],
    ) -> None:
        from pyapprox.probability.protocols.likelihood import (
            VectorizedLogLikelihoodProtocol,
        )

        if not isinstance(base_likelihood, VectorizedLogLikelihoodProtocol):
            raise TypeError(
                "base_likelihood must satisfy VectorizedLogLikelihoodProtocol, "
                f"got {type(base_likelihood).__name__}"
            )
        if observations.ndim != 2:
            raise ValueError(
                f"observations must be 2D (nobs, nexperiments), "
                f"got {observations.ndim}D"
            )
        self._base = base_likelihood
        self._observations = observations
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nobs(self) -> int:
        """Return the number of observations per experiment."""
        return self._observations.shape[0]

    def nexperiments(self) -> int:
        """Return the number of experiments."""
        return self._observations.shape[1]

    def logpdf(self, model_outputs: Array) -> Array:
        """
        Evaluate summed log-likelihood across all experiments.

        Parameters
        ----------
        model_outputs : Array
            Model predictions. Shape: ``(nobs, nsamples)``

        Returns
        -------
        Array
            Total log-likelihood per sample. Shape: ``(1, nsamples)``
        """
        # (nsamples, nexperiments)
        matrix = self._base.logpdf_vectorized(
            model_outputs, self._observations
        )
        # Sum across experiments → (nsamples,) → (1, nsamples)
        return self._bkd.reshape(
            self._bkd.sum(matrix, axis=1), (1, -1)
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MultiExperimentLogLikelihood("
            f"nobs={self.nobs()}, nexperiments={self.nexperiments()})"
        )

