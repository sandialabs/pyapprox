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

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)

from .linear_gaussian_model import LinearGaussianOEDModel


def _build_vandermonde(
    locations: Array,
    min_degree: int,
    degree: int,
    bkd: Backend[Array],
) -> Array:
    """
    Build polynomial Vandermonde matrix.

    Parameters
    ----------
    locations : Array
        Evaluation points. Shape: (n,)
    min_degree : int
        Minimum polynomial degree.
    degree : int
        Maximum polynomial degree.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    A : Array
        Vandermonde matrix. Shape: (n, degree - min_degree + 1)
        A[i, j] = locations[i]^(j + min_degree)
    """
    n = locations.shape[0]
    powers = bkd.arange(min_degree, degree + 1)
    x_col = bkd.reshape(locations, (n, 1))
    powers_row = bkd.reshape(powers, (1, len(powers)))
    return x_col ** powers_row


class LinearGaussianOEDBenchmark(Generic[Array]):
    """
    Linear Gaussian OED benchmark with analytical EIG.

    Sets up a polynomial regression problem where:
    - Design locations are in [-1, 1]
    - Forward model is polynomial basis evaluation
    - Prior and noise are isotropic Gaussian

    Uses composition with LinearGaussianOEDModel for shared logic.

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

        # Build Vandermonde design matrix
        locations = bkd.linspace(-1.0, 1.0, nobs)
        A = _build_vandermonde(locations, min_degree, degree, bkd)
        nparams = A.shape[1]

        # Build isotropic prior and noise
        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2

        # Create general model
        self._model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd, locations,
        )

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return self._model.nobs()

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._model.nparams()

    def noise_var(self) -> float:
        """Noise variance."""
        return self._noise_std ** 2

    def prior_var(self) -> float:
        """Prior variance."""
        return self._prior_std ** 2

    def noise_std(self) -> float:
        """Noise standard deviation."""
        return self._noise_std

    def prior_std(self) -> float:
        """Prior standard deviation."""
        return self._prior_std

    def design_matrix(self) -> Array:
        """Get the design matrix A. Shape: (nobs, nparams)"""
        return self._model.design_matrix()

    def design_locations(self) -> Array:
        """Get design locations. Shape: (nobs,)"""
        loc = self._model.design_locations()
        assert loc is not None
        return loc

    def model(self) -> LinearGaussianOEDModel[Array]:
        """Get the underlying LinearGaussianOEDModel."""
        return self._model

    def observation_model(self) -> FunctionFromCallable[Array]:
        """Return the observation model as a FunctionProtocol.

        Returns a callable mapping theta (nparams, nsamples) -> y (nobs, nsamples)
        via y = A @ theta.
        """
        return self._model.observation_model()

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
        return self._model.exact_eig(weights)

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
        return self._model.d_optimal_objective(weights)

    def prior_mean(self) -> Array:
        """Get prior mean (zero vector). Shape: (nparams, 1)"""
        return self._model.prior_mean()

    def prior_covariance(self) -> Array:
        """Get prior covariance (isotropic). Shape: (nparams, nparams)"""
        return self._model.prior_covariance()

    def noise_covariance(self) -> Array:
        """Get noise covariance (isotropic). Shape: (nobs, nobs)"""
        return self._model.noise_covariance()

    def noise_variances(self) -> Array:
        """Get noise variances for all observations. Shape: (nobs,)"""
        return self._model.noise_variances()

    def prior(self):
        """Return the prior as a Gaussian distribution."""
        return self._model.prior()

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
        return self._model.generate_observation_data(nsamples, seed)

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
        return self._model.generate_noisy_observations(nsamples, seed)

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
        return self._model.generate_latent_samples(nsamples, seed)


@BenchmarkRegistry.register(
    "linear_gaussian_oed",
    category="oed",
    description="Linear Gaussian OED benchmark with analytical EIG",
)
def _linear_gaussian_oed_factory(
    bkd: Backend[Array],
) -> LinearGaussianOEDBenchmark:
    return LinearGaussianOEDBenchmark(
        nobs=10, degree=3, noise_std=1.0, prior_std=1.0, bkd=bkd,
    )
