"""
Outer-data generators for fixed-measure OED objectives.

The outer expectation over data y is defined by fixed latent
coordinates; the realized nodes depend on the design weights because
the observation noise rescales with them. A generator owns exactly that
map: ``generator(design_weights) -> y_nodes`` with shape
``(nobs, nouter)``. Objectives consume a generator without knowing how
its coordinates were produced, so the same objective works with

- prior/noise draws in the double-loop pipeline's parameterization
  (:class:`ReparameterizedOuterData`), enabling exact outer-data
  sharing with pipeline estimators;
- marginal Gaussian coordinates through a Cholesky factor
  (:class:`MarginalOuterData`), which also covers correlated noise;
- any future deterministic quadrature construction.
"""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class ReparameterizedOuterData(Generic[Array]):
    """Diagonal-noise reparameterization: y = A theta + sqrt(s^2/w) eps.

    This is the double-loop pipeline's parameterization of the outer
    data, so passing a pipeline dataset's ``outer_shapes`` and
    ``latent_samples`` shares the outer data exactly.

    Parameters
    ----------
    outer_shapes : Array
        Noise-free model outputs A theta_k. Shape: (nobs, nouter)
    outer_latent : Array
        Standard-normal noise coordinates eps_k. Shape: (nobs, nouter)
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    bkd : Backend[Array]
    """

    def __init__(
        self,
        outer_shapes: Array,
        outer_latent: Array,
        noise_variances: Array,
        bkd: Backend[Array],
    ) -> None:
        if outer_shapes.shape != outer_latent.shape:
            raise ValueError(
                "outer_shapes and outer_latent must have equal shapes; "
                f"got {outer_shapes.shape} and {outer_latent.shape}"
            )
        self._outer_shapes = outer_shapes
        self._outer_latent = outer_latent
        self._noise_variances = noise_variances
        self._bkd = bkd

    @classmethod
    def from_prior_draws(
        cls,
        obs_mat: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_variances: Array,
        nouter: int,
        seed: int,
        bkd: Backend[Array],
    ) -> "ReparameterizedOuterData[Array]":
        """Draw theta from the prior and eps standard normal.

        The seed is part of the objective definition: equal seeds and
        sizes define identical outer measures.
        """
        rng = np.random.default_rng(seed)
        nparams = obs_mat.shape[1]
        nobs = obs_mat.shape[0]
        prior_chol = bkd.cholesky(prior_cov)
        xi = bkd.asarray(rng.standard_normal((nparams, nouter)))
        eps = bkd.asarray(rng.standard_normal((nobs, nouter)))
        outer_shapes = obs_mat @ (prior_mean + prior_chol @ xi)
        return cls(outer_shapes, eps, noise_variances, bkd)

    def nouter(self) -> int:
        return self._outer_shapes.shape[1]

    def __call__(self, design_weights: Array) -> Array:
        """Realize the data nodes at the given design weights.

        Parameters
        ----------
        design_weights : Array
            Shape: (nobs, 1)

        Returns
        -------
        Array
            Data nodes. Shape: (nobs, nouter)
        """
        bkd = self._bkd
        w = bkd.reshape(design_weights, (-1,))
        noise_std = bkd.sqrt(self._noise_variances / w)
        return (
            self._outer_shapes
            + bkd.reshape(noise_std, (-1, 1)) * self._outer_latent
        )


class MarginalOuterData(Generic[Array]):
    """Marginal Gaussian parameterization: y = A mu_0 + L(w) z.

    L(w) is the Cholesky factor of the prior predictive covariance
    A Sigma_0 A^T + Sigma_noise(w). More expensive per evaluation than
    :class:`ReparameterizedOuterData` (a factorization per call, with
    autograd through it) but extends to correlated noise covariances.

    Parameters
    ----------
    obs_mat : Array
        Observation matrix A. Shape: (nobs, nparams)
    prior_mean : Array
        Prior mean. Shape: (nparams, 1)
    prior_cov : Array
        Prior covariance. Shape: (nparams, nparams)
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    outer_nodes : Array
        Standard-normal coordinates z_k. Shape: (nobs, nouter)
    bkd : Backend[Array]
    """

    def __init__(
        self,
        obs_mat: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_variances: Array,
        outer_nodes: Array,
        bkd: Backend[Array],
    ) -> None:
        self._obs_mat = obs_mat
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._noise_variances = noise_variances
        self._outer_z = outer_nodes
        self._bkd = bkd

    @classmethod
    def from_seed(
        cls,
        obs_mat: Array,
        prior_mean: Array,
        prior_cov: Array,
        noise_variances: Array,
        nouter: int,
        seed: int,
        bkd: Backend[Array],
    ) -> "MarginalOuterData[Array]":
        """Draw the standard-normal coordinates from a seed."""
        rng = np.random.default_rng(seed)
        nobs = obs_mat.shape[0]
        z = bkd.asarray(rng.standard_normal((nobs, nouter)))
        return cls(obs_mat, prior_mean, prior_cov, noise_variances, z, bkd)

    def nouter(self) -> int:
        return self._outer_z.shape[1]

    def __call__(self, design_weights: Array) -> Array:
        """Realize the data nodes at the given design weights."""
        bkd = self._bkd
        w = bkd.reshape(design_weights, (-1,))
        noise_cov = bkd.diag(self._noise_variances / w)
        y_cov = (
            self._obs_mat @ (self._prior_cov @ self._obs_mat.T)
            + noise_cov
        )
        chol = bkd.cholesky(y_cov)
        return self._obs_mat @ self._prior_mean + chol @ self._outer_z
