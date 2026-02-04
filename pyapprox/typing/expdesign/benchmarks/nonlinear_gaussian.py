"""
Nonlinear Gaussian OED benchmark for prediction.

Provides a benchmark with a nonlinear QoI model (exp of linear model)
for testing prediction OED implementations. The linear observation model
combined with a nonlinear QoI allows for analytical computation of
expected utilities using lognormal distribution properties.

For the linear observation model:
    y = A @ theta + noise
where:
    - theta ~ N(0, prior_var * I) (prior)
    - noise ~ N(0, noise_var * I) (likelihood)

And the nonlinear QoI model:
    qoi = exp(B @ theta)
where B is the QoI design matrix.

The expected standard deviation of the lognormal QoI has an analytical
formula, allowing exact validation of numerical estimates.
"""

from typing import Generic

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend

from .linear_gaussian import LinearGaussianOEDBenchmark


class NonLinearGaussianOEDBenchmark(Generic[Array]):
    """
    Nonlinear Gaussian OED benchmark for prediction.

    Extends the linear Gaussian setup with:
    - Linear observation model: y = A @ theta + noise
    - Nonlinear QoI model: qoi = exp(B @ theta)

    The QoI is lognormal, enabling analytical utility computation.

    Parameters
    ----------
    nobs : int
        Number of observation locations.
    degree : int
        Maximum polynomial degree for observation model.
    noise_std : float
        Standard deviation of observation noise.
    prior_std : float
        Standard deviation of prior on coefficients.
    bkd : Backend[Array]
        Computational backend.
    npred : int
        Number of prediction locations. Default: 1.
    min_degree : int
        Minimum polynomial degree. Default: 0.

    Attributes
    ----------
    design_matrix : Array
        Observation model matrix A. Shape: (nobs, nparams)
    qoi_matrix : Array
        QoI model matrix B. Shape: (npred, nparams)
    """

    def __init__(
        self,
        nobs: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        bkd: Backend[Array],
        npred: int = 1,
        min_degree: int = 0,
    ) -> None:
        self._bkd = bkd
        self._nobs = nobs
        self._degree = degree
        self._min_degree = min_degree
        self._noise_std = noise_std
        self._prior_std = prior_std
        self._npred = npred

        # Setup observation model (same as linear benchmark)
        self._design_locations = self._setup_design_locations()
        self._design_matrix = self._setup_design_matrix()
        self._nparams = self._design_matrix.shape[1]

        # Setup QoI model
        self._qoi_locations = self._setup_qoi_locations()
        self._qoi_matrix = self._setup_qoi_matrix()
        self._qoi_quad_weights = bkd.full((npred, 1), 1.0 / npred)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._nparams

    def npred(self) -> int:
        """Number of prediction QoI locations."""
        return self._npred

    def noise_var(self) -> float:
        """Noise variance."""
        return self._noise_std ** 2

    def noise_std(self) -> float:
        """Noise standard deviation."""
        return self._noise_std

    def prior_var(self) -> float:
        """Prior variance."""
        return self._prior_std ** 2

    def prior_std(self) -> float:
        """Prior standard deviation."""
        return self._prior_std

    def design_matrix(self) -> Array:
        """Get the observation design matrix A. Shape: (nobs, nparams)"""
        return self._design_matrix

    def design_locations(self) -> Array:
        """Get observation design locations. Shape: (nobs,)"""
        return self._design_locations

    def qoi_matrix(self) -> Array:
        """Get the QoI design matrix B. Shape: (npred, nparams)"""
        return self._qoi_matrix

    def qoi_locations(self) -> Array:
        """Get QoI prediction locations. Shape: (npred,)"""
        return self._qoi_locations

    def qoi_quad_weights(self) -> Array:
        """Get QoI quadrature weights. Shape: (npred, 1)"""
        return self._qoi_quad_weights

    def _setup_design_locations(self) -> Array:
        """Set up observation locations in [-1, 1]."""
        return self._bkd.linspace(-1.0, 1.0, self._nobs)

    def _setup_qoi_locations(self) -> Array:
        """Set up QoI prediction locations in [-2/3, 2/3]."""
        return self._bkd.linspace(-2.0 / 3.0, 2.0 / 3.0, self._npred)

    def _setup_design_matrix(self) -> Array:
        """
        Set up polynomial design matrix for observations.

        Returns
        -------
        A : Array
            Design matrix. Shape: (nobs, nparams)
            A[i, j] = x_i^(j + min_degree)
        """
        x = self._design_locations
        powers = self._bkd.arange(self._min_degree, self._degree + 1)
        x_col = self._bkd.reshape(x, (self._nobs, 1))
        powers_row = self._bkd.reshape(powers, (1, len(powers)))
        return x_col ** powers_row

    def _setup_qoi_matrix(self) -> Array:
        """
        Set up polynomial design matrix for QoI predictions.

        Returns
        -------
        B : Array
            QoI matrix. Shape: (npred, nparams)
            B[i, j] = qoi_x_i^(j + min_degree)
        """
        x = self._qoi_locations
        powers = self._bkd.arange(self._min_degree, self._degree + 1)
        x_col = self._bkd.reshape(x, (self._npred, 1))
        powers_row = self._bkd.reshape(powers, (1, len(powers)))
        return x_col ** powers_row

    def generate_parameter_samples(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> Array:
        """
        Generate parameter samples from the prior.

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
        theta_np = self._prior_std * np.random.randn(self._nparams, nsamples)
        return self._bkd.asarray(theta_np)

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
            Observations. Shape: (nobs, nsamples)
        """
        theta_samples = self.generate_parameter_samples(nsamples, seed)
        y_samples = self._bkd.dot(self._design_matrix, theta_samples)
        return theta_samples, y_samples

    def generate_qoi_data(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array, Array]:
        """
        Generate QoI values from the nonlinear (exponential) model.

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
        linear_qoi : Array
            Linear QoI values (B @ theta). Shape: (npred, nsamples)
        exp_qoi : Array
            Exponential QoI values (exp(B @ theta)). Shape: (npred, nsamples)
        """
        theta_samples = self.generate_parameter_samples(nsamples, seed)
        linear_qoi = self._bkd.dot(self._qoi_matrix, theta_samples)
        exp_qoi = self._bkd.exp(linear_qoi)
        return theta_samples, linear_qoi, exp_qoi

    def generate_noisy_observations(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array, Array]:
        """
        Generate noisy observations.

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
        noise_np = self._noise_std * np.random.randn(self._nobs, nsamples)
        noise = self._bkd.asarray(noise_np)
        y_noisy = y_clean + noise

        return theta_samples, y_clean, y_noisy

    def generate_latent_samples(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> Array:
        """
        Generate latent noise samples for reparameterization trick.

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

    def noise_variances(self) -> Array:
        """
        Get noise variances for all observations.

        Returns
        -------
        variances : Array
            Noise variances. Shape: (nobs,)
        """
        return self._bkd.full((self._nobs,), self.noise_var())

    def prior_mean(self) -> Array:
        """
        Get prior mean (zero vector).

        Returns
        -------
        mean : Array
            Prior mean. Shape: (nparams, 1)
        """
        return self._bkd.zeros((self._nparams, 1))

    def prior_covariance(self) -> Array:
        """
        Get prior covariance (isotropic).

        Returns
        -------
        cov : Array
            Prior covariance. Shape: (nparams, nparams)
        """
        return self._bkd.eye(self._nparams) * self.prior_var()
