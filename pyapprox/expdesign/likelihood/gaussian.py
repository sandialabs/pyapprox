"""
Gaussian OED likelihood implementations.

Provides OED-specific likelihood wrappers for Gaussian noise models,
including vectorized evaluation and Jacobians w.r.t. design weights.
"""

import math
from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.likelihood.dispatch import (
    get_logpdf_matrix_impl,
    get_jacobian_matrix_impl,
    get_evidence_jacobian_impl,
)


class GaussianOEDOuterLoopLikelihood(Generic[Array]):
    """
    Outer loop likelihood for Gaussian OED.

    Evaluates log p(obs | theta, weights) where obs and shapes (model outputs)
    have the same shape (nobs, nouter).

    The design weights scale the noise variance:
        effective_variance[i] = base_variance[i] / weights[i]

    Parameters
    ----------
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._base_variances = noise_variances
        self._nobs = noise_variances.shape[0]

        # State
        self._shapes: Optional[Array] = None
        self._observations: Optional[Array] = None
        self._latent_samples: Optional[Array] = None
        self._design_weights: Optional[Array] = None
        self._residuals: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def nouter(self) -> int:
        """Number of outer samples."""
        if self._shapes is None:
            raise ValueError("Must call set_shapes first")
        return self._shapes.shape[1]

    def set_shapes(self, shapes: Array) -> None:
        """
        Set model outputs (means for Gaussian).

        Parameters
        ----------
        shapes : Array
            Model outputs. Shape: (nobs, nouter)
        """
        if shapes.ndim != 2:
            raise ValueError(f"shapes must be 2D, got {shapes.ndim}D")
        if shapes.shape[0] != self._nobs:
            raise ValueError(
                f"shapes has wrong number of observations: "
                f"expected {self._nobs}, got {shapes.shape[0]}"
            )
        self._shapes = shapes
        self._residuals = None  # Invalidate cached residuals

    def set_observations(self, obs: Array) -> None:
        """
        Set artificial observations.

        Parameters
        ----------
        obs : Array
            Observations. Shape: (nobs, nouter) - must match shapes
        """
        if self._shapes is None:
            raise ValueError("Must call set_shapes first")
        if obs.shape != self._shapes.shape:
            raise ValueError(
                f"obs shape {obs.shape} must match shapes {self._shapes.shape}"
            )
        self._observations = obs
        self._residuals = obs - self._shapes

    def set_latent_samples(self, latent_samples: Array) -> None:
        """
        Set latent samples for reparameterization trick.

        Parameters
        ----------
        latent_samples : Array
            Standard normal samples. Shape: (nobs, nouter)
        """
        if self._shapes is None:
            raise ValueError("Must call set_shapes first")
        if latent_samples.shape != self._shapes.shape:
            raise ValueError(
                f"latent_samples shape {latent_samples.shape} must match "
                f"shapes {self._shapes.shape}"
            )
        self._latent_samples = latent_samples

    def set_design_weights(self, weights: Array) -> None:
        """
        Set design weights.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)
        """
        if weights.shape != (self._nobs, 1):
            raise ValueError(
                f"weights must have shape ({self._nobs}, 1), got {weights.shape}"
            )
        self._design_weights = weights

    def _compute_weighted_inv_variance(self) -> Array:
        """Compute inverse of weighted variance: weights / base_variance."""
        if self._design_weights is None:
            raise ValueError("Must call set_design_weights first")
        # effective_var = base_var / weights
        # inv_effective_var = weights / base_var
        return self._design_weights[:, 0] / self._base_variances

    def _compute_log_normalization(self) -> Array:
        """Compute log normalization constant with current weights.

        Returns a scalar Array to preserve the PyTorch autograd computation graph.
        """
        if self._design_weights is None:
            raise ValueError("Must call set_design_weights first")
        # log|Cov| = sum(log(base_var / weights)) = sum(log(base_var)) - sum(log(weights))
        log_det = (
            self._bkd.sum(self._bkd.log(self._base_variances))
            - self._bkd.sum(self._bkd.log(self._design_weights))
        )
        const = self._bkd.asarray(-0.5 * self._nobs * math.log(2 * math.pi))
        return const - 0.5 * log_det

    def __call__(self, design_weights: Array) -> Array:
        """
        Evaluate log-likelihood for all outer samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-likelihood values. Shape: (1, nouter)
        """
        self.set_design_weights(design_weights)

        if self._residuals is None:
            raise ValueError("Must call set_shapes and set_observations first")

        inv_var = self._compute_weighted_inv_variance()
        log_norm = self._compute_log_normalization()

        # Squared Mahalanobis distance: sum_i (residual_i^2 * inv_var_i)
        squared_dist = self._bkd.sum(
            self._residuals**2 * inv_var[:, None], axis=0
        )

        result = log_norm - 0.5 * squared_dist
        return self._bkd.reshape(result, (1, -1))

    def jacobian(self, design_weights: Array) -> Array:
        """
        Jacobian of log-likelihood w.r.t. design weights.

        The Jacobian has two components:
        1. Quadratic term: d/dw (-0.5 * r^2 / (var/w)) = 0.5 * r^2 * w / var^2 * (-var/w^2)
                                                       = -0.5 * r^2 / (var * w)
        2. Determinant term: d/dw (-0.5 * log(var/w)) = 0.5 / w
        3. Reparameterization term (if using reparam trick)

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nouter, nobs)
        """
        self.set_design_weights(design_weights)

        if self._residuals is None:
            raise ValueError("Must call set_shapes and set_observations first")

        inv_var = self._compute_weighted_inv_variance()

        # Component 1: Quadratic term
        # d/dw_k log p = d/dw_k (-0.5 * sum_i r_i^2 * w_i / var_i)
        #              = -0.5 * r_k^2 / var_k  (for each observation)
        jac = -0.5 * self._residuals.T**2 / self._base_variances

        # Component 2: Determinant term
        # d/dw_k (-0.5 * log|Cov|) = d/dw_k (0.5 * sum_i log(w_i)) = 0.5 / w_k
        jac = jac + 0.5 / design_weights.T

        # Component 3: Reparameterization term (if latent samples set)
        if self._latent_samples is not None:
            # obs = shapes + sqrt(var/w) * latent
            # d(obs)/dw = -0.5 * sqrt(var) * latent / w^{3/2}
            # d(residual)/dw = d(obs)/dw = -0.5 * sqrt(var) * latent / w^{3/2}
            # Chain rule: d/dw (-0.5 * r^2 * w/var) = -r * w/var * d(r)/dw
            #           = -r * w/var * (-0.5 * sqrt(var) * latent / w^{3/2})
            #           = 0.5 * r * latent / (sqrt(var) * sqrt(w))
            jac = jac + 0.5 * (
                self._residuals.T
                * self._latent_samples.T
                / (self._bkd.sqrt(self._base_variances) * self._bkd.sqrt(design_weights.T))
            )

        return jac


class GaussianOEDInnerLoopLikelihood(Generic[Array]):
    """
    Inner loop likelihood for Gaussian OED.

    Computes the full (ninner, nouter) log-likelihood matrix where:
    - ninner: number of prior samples (shapes)
    - nouter: number of observation realizations

    Entry [i, j] = log p(obs_j | theta_i, weights)

    Parameters
    ----------
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        noise_variances: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        self._base_variances = noise_variances
        self._nobs = noise_variances.shape[0]

        # Dispatch implementations (auto-selected based on backend type)
        self._logpdf_impl = get_logpdf_matrix_impl(bkd)
        self._jacobian_impl = get_jacobian_matrix_impl(bkd)
        self._evidence_jacobian_impl = get_evidence_jacobian_impl(bkd)

        # State
        self._shapes: Optional[Array] = None  # (nobs, ninner)
        self._observations: Optional[Array] = None  # (nobs, nouter)
        self._latent_samples: Optional[Array] = None  # (nobs, nouter)
        self._design_weights: Optional[Array] = None  # (nobs, 1)

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def ninner(self) -> int:
        """Number of inner (prior) samples."""
        if self._shapes is None:
            raise ValueError("Must call set_shapes first")
        return self._shapes.shape[1]

    def nouter(self) -> int:
        """Number of outer (observation) samples."""
        if self._observations is None:
            raise ValueError("Must call set_observations first")
        return self._observations.shape[1]

    def set_shapes(self, shapes: Array) -> None:
        """
        Set model outputs for inner samples.

        Parameters
        ----------
        shapes : Array
            Model outputs. Shape: (nobs, ninner)
        """
        if shapes.ndim != 2:
            raise ValueError(f"shapes must be 2D, got {shapes.ndim}D")
        if shapes.shape[0] != self._nobs:
            raise ValueError(
                f"shapes has wrong number of observations: "
                f"expected {self._nobs}, got {shapes.shape[0]}"
            )
        self._shapes = shapes

    def set_observations(self, obs: Array) -> None:
        """
        Set artificial observations for outer samples.

        Parameters
        ----------
        obs : Array
            Observations. Shape: (nobs, nouter)
        """
        if obs.ndim != 2:
            raise ValueError(f"obs must be 2D, got {obs.ndim}D")
        if obs.shape[0] != self._nobs:
            raise ValueError(
                f"obs has wrong number of observations: "
                f"expected {self._nobs}, got {obs.shape[0]}"
            )
        self._observations = obs

    def set_latent_samples(self, latent_samples: Array) -> None:
        """
        Set latent samples for reparameterization trick.

        Parameters
        ----------
        latent_samples : Array
            Standard normal samples. Shape: (nobs, nouter)
        """
        if self._observations is None:
            raise ValueError("Must call set_observations first")
        if latent_samples.shape != self._observations.shape:
            raise ValueError(
                f"latent_samples shape {latent_samples.shape} must match "
                f"observations {self._observations.shape}"
            )
        self._latent_samples = latent_samples

    def set_design_weights(self, weights: Array) -> None:
        """
        Set design weights.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)
        """
        if weights.shape != (self._nobs, 1):
            raise ValueError(
                f"weights must have shape ({self._nobs}, 1), got {weights.shape}"
            )
        self._design_weights = weights

    def logpdf_matrix(self, design_weights: Array) -> Array:
        """
        Compute full log-likelihood matrix.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Log-likelihood matrix. Shape: (ninner, nouter)
        """
        self.set_design_weights(design_weights)
        if self._shapes is None or self._observations is None:
            raise ValueError("Must call set_shapes and set_observations first")

        return self._logpdf_impl(
            self._shapes, self._observations, self._base_variances,
            design_weights, self._bkd,
        )

    def jacobian_matrix(self, design_weights: Array) -> Array:
        """
        Jacobian of log-likelihood matrix w.r.t. design weights.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        Array
            Jacobian tensor. Shape: (ninner, nouter, nobs)
        """
        self.set_design_weights(design_weights)
        if self._shapes is None or self._observations is None:
            raise ValueError("Must call set_shapes and set_observations first")

        return self._jacobian_impl(
            self._shapes, self._observations, self._latent_samples,
            self._base_variances, design_weights, self._bkd,
        )

    def evidence_jacobian(
        self,
        design_weights: Array,
        quad_weighted_like: Array,
    ) -> Array:
        """
        Compute fused evidence jacobian.

        When Numba is active, this avoids materializing the full
        (ninner, nouter, nobs) jacobian tensor by fusing the jacobian
        computation with the einsum contraction.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)
        quad_weighted_like : Array
            Quadrature-weighted likelihoods. Shape: (ninner, nouter)

        Returns
        -------
        Array
            Evidence jacobian. Shape: (nouter, nobs)
        """
        self.set_design_weights(design_weights)
        if self._shapes is None or self._observations is None:
            raise ValueError("Must call set_shapes and set_observations first")

        return self._evidence_jacobian_impl(
            self._shapes, self._observations, self._latent_samples,
            self._base_variances, design_weights, quad_weighted_like,
            self._bkd,
        )

    def create_outer_loop_likelihood(self) -> GaussianOEDOuterLoopLikelihood[Array]:
        """
        Create a paired outer loop likelihood.

        Returns
        -------
        GaussianOEDOuterLoopLikelihood
            Outer loop likelihood with same noise model.
        """
        return GaussianOEDOuterLoopLikelihood(self._base_variances, self._bkd)
