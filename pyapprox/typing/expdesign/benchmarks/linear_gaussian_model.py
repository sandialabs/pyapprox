"""
General linear Gaussian OED model.

Holds an arbitrary design matrix A, full prior covariance, and full noise
covariance. Provides exact expected information gain (EIG) via conjugate
Gaussian posterior, data generation via Cholesky decomposition, and
D-optimal objective.

For the linear Gaussian model:
    y = A @ theta + noise
where:
    - theta ~ N(prior_mean, prior_covariance)
    - noise ~ N(0, noise_covariance)

This generalizes the isotropic case to support non-isotropic priors and
heteroscedastic noise.

References
----------
Alexanderian, A. and Saibaba, A.K.
"Efficient D-Optimal Design of Experiments for Infinite-Dimensional
Bayesian Linear Inverse Problems"
SIAM Journal on Scientific Computing 2018 40:5, A2956-A2985
"""

from typing import Generic, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.inverse.conjugate.gaussian import (
    DenseGaussianConjugatePosterior,
)
from pyapprox.typing.probability.gaussian import (
    DenseCholeskyMultivariateGaussian,
)


class LinearGaussianOEDModel(Generic[Array]):
    """
    General linear Gaussian OED model with analytical EIG.

    Supports arbitrary design matrices, full prior covariance, and full
    noise covariance. Computes exact EIG using the conjugate Gaussian
    posterior.

    For the isotropic case (prior_cov = s^2 I, noise_cov = n^2 I), this
    reduces to the classical formula:
        EIG(w) = 1/2 * log(det(I + A^T diag(w) A * s^2 / n^2))

    Parameters
    ----------
    design_matrix : Array
        Forward model matrix A. Shape: (nobs, nparams)
    prior_mean : Array
        Prior mean. Shape: (nparams, 1)
    prior_covariance : Array
        Prior covariance. Shape: (nparams, nparams)
    noise_covariance : Array
        Noise covariance. Shape: (nobs, nobs)
    bkd : Backend[Array]
        Computational backend.
    design_locations : Array or None, optional
        Physical locations of observation sensors. Shape: (nobs,)
    """

    def __init__(
        self,
        design_matrix: Array,
        prior_mean: Array,
        prior_covariance: Array,
        noise_covariance: Array,
        bkd: Backend[Array],
        design_locations: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._design_matrix = design_matrix
        self._nobs, self._nparams = design_matrix.shape

        # Validate shapes
        if prior_mean.shape != (self._nparams, 1):
            raise ValueError(
                f"prior_mean has wrong shape {prior_mean.shape}, "
                f"expected ({self._nparams}, 1)"
            )
        self._prior_mean = prior_mean

        if prior_covariance.shape != (self._nparams, self._nparams):
            raise ValueError(
                f"prior_covariance has wrong shape {prior_covariance.shape}, "
                f"expected ({self._nparams}, {self._nparams})"
            )
        self._prior_covariance = prior_covariance

        if noise_covariance.shape != (self._nobs, self._nobs):
            raise ValueError(
                f"noise_covariance has wrong shape {noise_covariance.shape}, "
                f"expected ({self._nobs}, {self._nobs})"
            )
        self._noise_covariance = noise_covariance

        if design_locations is not None and design_locations.shape != (self._nobs,):
            raise ValueError(
                f"design_locations has wrong shape {design_locations.shape}, "
                f"expected ({self._nobs},)"
            )
        self._design_locations = design_locations

        # Cache Cholesky factors for data generation
        self._L_prior = bkd.cholesky(prior_covariance)
        self._L_noise = bkd.cholesky(noise_covariance)

        # Cache diagonal of noise covariance
        self._noise_diag = bkd.diag(noise_covariance)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._nparams

    def design_matrix(self) -> Array:
        """Get the design matrix A. Shape: (nobs, nparams)"""
        return self._design_matrix

    def design_locations(self) -> Optional[Array]:
        """Get design locations. Shape: (nobs,) or None"""
        return self._design_locations

    def prior_mean(self) -> Array:
        """Get prior mean. Shape: (nparams, 1)"""
        return self._prior_mean

    def prior_covariance(self) -> Array:
        """Get prior covariance. Shape: (nparams, nparams)"""
        return self._prior_covariance

    def noise_covariance(self) -> Array:
        """Get noise covariance. Shape: (nobs, nobs)"""
        return self._noise_covariance

    def noise_variances(self) -> Array:
        """Get diagonal of noise covariance. Shape: (nobs,)"""
        return self._noise_diag

    def prior(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """Return the prior as a Gaussian distribution."""
        return DenseCholeskyMultivariateGaussian(
            self._prior_mean, self._prior_covariance, self._bkd,
        )

    def exact_eig(self, weights: Array) -> float:
        """
        Compute exact expected information gain.

        Uses DenseGaussianConjugatePosterior with weighted noise covariance.
        Zero-weight sensors are excluded.

        For the isotropic case, this reduces to:
            EIG(w) = 1/2 * log(det(I + A^T diag(w) A * prior_var / noise_var))

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        eig : float
            Exact expected information gain.
        """
        bkd = self._bkd
        w = bkd.reshape(weights, (self._nobs,))

        # Identify active sensors (positive weight)
        w_np = bkd.to_numpy(w)
        active_mask = w_np > 0
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        if n_active == 0:
            return 0.0

        # Extract active rows of design matrix
        A_active = self._design_matrix[active_indices, :]

        # Build effective noise covariance: diag(noise_var_i / w_i) for active
        noise_diag_np = bkd.to_numpy(self._noise_diag)
        w_active_np = w_np[active_mask]
        noise_diag_eff_np = noise_diag_np[active_mask] / w_active_np
        noise_cov_eff = bkd.diag(bkd.asarray(noise_diag_eff_np))

        # Compute EIG via conjugate posterior
        post = DenseGaussianConjugatePosterior(
            A_active,
            self._prior_mean,
            self._prior_covariance,
            noise_cov_eff,
            bkd,
        )
        # EIG is data-independent for linear Gaussian; use dummy observations
        dummy_obs = bkd.zeros((n_active, 1))
        post.compute(dummy_obs)
        return post.expected_kl_divergence()

    def d_optimal_objective(self, weights: Array) -> float:
        """
        Compute D-optimal objective (negative EIG).

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        obj : float
            D-optimal objective = -EIG
        """
        return -self.exact_eig(weights)

    def generate_parameter_samples(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> Array:
        """
        Generate parameter samples from the prior.

        Uses Cholesky decomposition: theta = L_prior @ z + prior_mean
        where z ~ N(0, I). For the isotropic case (prior_cov = s^2 I),
        L_prior = s * I, so this is identical to s * z + 0 = s * randn.

        Parameters
        ----------
        nsamples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        theta_samples : Array
            Parameter samples. Shape: (nparams, nsamples)
        """
        np.random.seed(seed)
        z_np = np.random.randn(self._nparams, nsamples)
        z = self._bkd.asarray(z_np)
        return self._bkd.dot(self._L_prior, z) + self._prior_mean

    def generate_observation_data(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array]:
        """
        Generate noiseless observations from the linear model.

        Parameters
        ----------
        nsamples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        theta_samples : Array
            Parameter samples. Shape: (nparams, nsamples)
        y_samples : Array
            Observations (noiseless). Shape: (nobs, nsamples)
        """
        theta_samples = self.generate_parameter_samples(nsamples, seed)
        y_samples = self._bkd.dot(self._design_matrix, theta_samples)
        return theta_samples, y_samples

    def generate_noisy_observations(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array, Array]:
        """
        Generate noisy observations.

        Uses Cholesky decomposition: noise = L_noise @ z
        where z ~ N(0, I). For isotropic noise (noise_cov = n^2 I),
        L_noise = n * I, so noise = n * randn.

        Parameters
        ----------
        nsamples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        theta_samples : Array
            Parameter samples. Shape: (nparams, nsamples)
        y_clean : Array
            Noiseless observations. Shape: (nobs, nsamples)
        y_noisy : Array
            Noisy observations. Shape: (nobs, nsamples)
        """
        theta_samples, y_clean = self.generate_observation_data(nsamples, seed)

        # Add noise with different seed
        np.random.seed(seed + 1000)
        z_np = np.random.randn(self._nobs, nsamples)
        z = self._bkd.asarray(z_np)
        noise = self._bkd.dot(self._L_noise, z)
        y_noisy = y_clean + noise

        return theta_samples, y_clean, y_noisy

    def generate_latent_samples(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> Array:
        """
        Generate latent noise samples for reparameterization trick.

        Returns standard normal samples (unscaled).

        Parameters
        ----------
        nsamples : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        -------
        latent : Array
            Standard normal samples. Shape: (nobs, nsamples)
        """
        np.random.seed(seed + 2000)
        latent_np = np.random.randn(self._nobs, nsamples)
        return self._bkd.asarray(latent_np)
