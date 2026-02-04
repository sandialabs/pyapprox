"""
Linear Gaussian OED benchmark problem.

Provides a benchmark problem with analytical expected information gain (EIG)
for validating OED implementations.

For the linear Gaussian model:
    y = A @ theta + noise
where:
    - theta ~ N(0, prior_var * I) (prior)
    - noise ~ N(0, noise_var * I) (likelihood)

The analytical EIG is:
    EIG(w) = 1/2 * log(det(I + A^T @ diag(w) @ A * prior_var / noise_var))

This provides ground truth for testing numerical EIG estimates.

References
----------
Alexanderian, A. and Saibaba, A.K.
"Efficient D-Optimal Design of Experiments for Infinite-Dimensional
Bayesian Linear Inverse Problems"
SIAM Journal on Scientific Computing 2018 40:5, A2956-A2985
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class LinearGaussianOEDBenchmark(Generic[Array]):
    """
    Linear Gaussian OED benchmark with analytical EIG.

    Sets up a polynomial regression problem where:
    - Design locations are in [-1, 1]
    - Forward model is polynomial basis evaluation
    - Prior and noise are isotropic Gaussian

    Parameters
    ----------
    nobs : int
        Number of observation locations.
    degree : int
        Maximum polynomial degree.
    noise_std : float
        Standard deviation of observation noise.
    prior_std : float
        Standard deviation of prior on coefficients.
    bkd : Backend[Array]
        Computational backend.
    min_degree : int, optional
        Minimum polynomial degree (default 0).

    Attributes
    ----------
    design_matrix : Array
        Forward model matrix A. Shape: (nobs, nparams)
    noise_var : float
        Noise variance (noise_std^2).
    prior_var : float
        Prior variance (prior_std^2).
    """

    def __init__(
        self,
        nobs: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        bkd: Backend[Array],
        min_degree: int = 0,
    ) -> None:
        self._bkd = bkd
        self._nobs = nobs
        self._degree = degree
        self._min_degree = min_degree
        self._noise_std = noise_std
        self._prior_std = prior_std

        # Setup problem components
        self._design_locations = self._setup_design_locations()
        self._design_matrix = self._setup_design_matrix()
        self._nparams = self._design_matrix.shape[1]

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._nparams

    def noise_var(self) -> float:
        """Noise variance."""
        return self._noise_std ** 2

    def prior_var(self) -> float:
        """Prior variance."""
        return self._prior_std ** 2

    def design_matrix(self) -> Array:
        """Get the design matrix A. Shape: (nobs, nparams)"""
        return self._design_matrix

    def design_locations(self) -> Array:
        """Get design locations. Shape: (nobs,)"""
        return self._design_locations

    def _setup_design_locations(self) -> Array:
        """Set up observation locations in [-1, 1]."""
        return self._bkd.linspace(-1.0, 1.0, self._nobs)

    def _setup_design_matrix(self) -> Array:
        """
        Set up polynomial design matrix.

        Returns
        -------
        A : Array
            Design matrix. Shape: (nobs, nparams)
            A[i, j] = x_i^(j + min_degree)
        """
        x = self._design_locations
        powers = self._bkd.arange(self._min_degree, self._degree + 1)
        # Vandermonde-like matrix: each row is [x^0, x^1, ..., x^degree]
        # Use broadcasting: x[:, None] ** powers[None, :]
        x_col = self._bkd.reshape(x, (self._nobs, 1))
        powers_row = self._bkd.reshape(powers, (1, len(powers)))
        return x_col ** powers_row

    def exact_eig(self, weights: Array) -> float:
        """
        Compute exact expected information gain.

        For linear Gaussian model with isotropic prior and noise:
            EIG(w) = 1/2 * log(det(I + A^T @ diag(w) @ A * prior_var / noise_var))

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        eig : float
            Exact expected information gain.
        """
        A = self._design_matrix
        ratio = self._prior_std ** 2 / self._noise_std ** 2

        # Compute A^T @ diag(w) @ A
        # = A^T @ (w * A) where w is broadcast
        w = self._bkd.reshape(weights, (self._nobs,))
        AtWA = self._bkd.dot(A.T, w[:, None] * A)

        # Y = I + AtWA * ratio
        Y = self._bkd.eye(self._nparams) + AtWA * ratio

        # EIG = 1/2 * log(det(Y))
        sign, logdet = self._bkd.slogdet(Y)
        return 0.5 * float(self._bkd.to_numpy(logdet))

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

    def generate_data(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array]:
        """
        Generate synthetic data from the model.

        Parameters
        ----------
        nsamples : int
            Number of parameter samples to generate.
        seed : int
            Random seed.

        Returns
        -------
        theta_samples : Array
            Parameter samples from prior. Shape: (nparams, nsamples)
        y_samples : Array
            Observations (noiseless). Shape: (nobs, nsamples)
        """
        np.random.seed(seed)

        # Sample from prior: theta ~ N(0, prior_var * I)
        theta_np = self._prior_std * np.random.randn(self._nparams, nsamples)
        theta_samples = self._bkd.asarray(theta_np)

        # Forward model: y = A @ theta
        y_samples = self._bkd.dot(self._design_matrix, theta_samples)

        return theta_samples, y_samples

    def generate_noisy_data(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array, Array]:
        """
        Generate synthetic data with noise.

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
        theta_samples, y_clean = self.generate_data(nsamples, seed)

        # Add noise
        np.random.seed(seed + 1000)  # Different seed for noise
        noise_np = self._noise_std * np.random.randn(self._nobs, nsamples)
        noise = self._bkd.asarray(noise_np)
        y_noisy = y_clean + noise

        return theta_samples, y_clean, y_noisy
