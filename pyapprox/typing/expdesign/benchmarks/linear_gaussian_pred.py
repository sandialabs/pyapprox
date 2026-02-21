"""
Linear Gaussian OED benchmark for prediction with linear QoI.

Provides a benchmark with a linear QoI model (B @ theta) for testing
prediction OED implementations. The linear observation and QoI models
combined with Gaussian priors and noise allow for analytical computation
of expected utilities using conjugate Gaussian formulas.

For the linear observation model:
    y = A @ theta + noise
where:
    - theta ~ N(0, prior_var * I) (prior)
    - noise ~ N(0, noise_var * I) (likelihood)

And the linear QoI model:
    qoi = B @ theta
where B is the QoI design matrix.

The expected standard deviation of the Gaussian QoI has an analytical
formula, allowing exact validation of numerical estimates.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend

from .linear_gaussian import _build_vandermonde
from .linear_gaussian_model import LinearGaussianOEDModel


class LinearGaussianPredOEDBenchmark(Generic[Array]):
    """
    Linear Gaussian OED benchmark for prediction with linear QoI.

    Extends the linear Gaussian setup with:
    - Linear observation model: y = A @ theta + noise
    - Linear QoI model: qoi = B @ theta

    The QoI is Gaussian, enabling analytical utility computation via
    ConjugateGaussianOEDExpectedStdDev and ConjugateGaussianOEDExpectedAVaRDev.

    Uses composition with LinearGaussianOEDModel for observation model logic.

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
    npred : int
        Number of prediction locations.
    bkd : Backend[Array]
        Computational backend.
    min_degree : int
        Minimum polynomial degree. Default: 0.
    """

    def __init__(
        self,
        nobs: int,
        degree: int,
        noise_std: float,
        prior_std: float,
        npred: int,
        bkd: Backend[Array],
        min_degree: int = 0,
    ) -> None:
        self._bkd = bkd
        self._nobs = nobs
        self._degree = degree
        self._min_degree = min_degree
        self._noise_std = noise_std
        self._prior_std = prior_std
        self._npred = npred

        # Build observation model via composition
        obs_locations = bkd.linspace(-1.0, 1.0, nobs)
        A = _build_vandermonde(obs_locations, min_degree, degree, bkd)
        nparams = A.shape[1]

        prior_mean = bkd.zeros((nparams, 1))
        prior_cov = bkd.eye(nparams) * prior_std ** 2
        noise_cov = bkd.eye(nobs) * noise_std ** 2

        self._model = LinearGaussianOEDModel(
            A, prior_mean, prior_cov, noise_cov, bkd, obs_locations,
        )

        # QoI-specific setup: linear QoI = B @ theta
        self._qoi_locations = bkd.linspace(-2.0 / 3.0, 2.0 / 3.0, npred)
        self._qoi_matrix = _build_vandermonde(
            self._qoi_locations, min_degree, degree, bkd,
        )
        self._qoi_quad_weights = bkd.full((npred, 1), 1.0 / npred)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def nobs(self) -> int:
        """Number of observations."""
        return self._model.nobs()

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._model.nparams()

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
        return self._model.design_matrix()

    def design_locations(self) -> Array:
        """Get observation design locations. Shape: (nobs,)"""
        loc = self._model.design_locations()
        assert loc is not None
        return loc

    def model(self) -> LinearGaussianOEDModel[Array]:
        """Get the underlying LinearGaussianOEDModel."""
        return self._model

    def qoi_matrix(self) -> Array:
        """Get the QoI design matrix B. Shape: (npred, nparams)"""
        return self._qoi_matrix

    def qoi_locations(self) -> Array:
        """Get QoI prediction locations. Shape: (npred,)"""
        return self._qoi_locations

    def qoi_quad_weights(self) -> Array:
        """Get QoI quadrature weights. Shape: (npred, 1)"""
        return self._qoi_quad_weights

    def noise_variances(self) -> Array:
        """Get noise variances for all observations. Shape: (nobs,)"""
        return self._model.noise_variances()

    def prior_mean(self) -> Array:
        """Get prior mean (zero vector). Shape: (nparams, 1)"""
        return self._model.prior_mean()

    def prior_covariance(self) -> Array:
        """Get prior covariance (isotropic). Shape: (nparams, nparams)"""
        return self._model.prior_covariance()

    def prior(self):
        """Return the prior as a Gaussian distribution."""
        return self._model.prior()

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
        return self._model.generate_parameter_samples(nsamples, seed)

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
        return self._model.generate_observation_data(nsamples, seed)

    def generate_qoi_data(
        self,
        nsamples: int,
        seed: int = 42,
    ) -> tuple[Array, Array]:
        """
        Generate QoI values from the linear model: qoi = B @ theta.

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
        """
        theta_samples = self.generate_parameter_samples(nsamples, seed)
        linear_qoi = self._bkd.dot(self._qoi_matrix, theta_samples)
        return theta_samples, linear_qoi

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
