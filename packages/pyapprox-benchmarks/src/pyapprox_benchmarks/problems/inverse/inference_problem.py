"""Bayesian inference problem (obs_map + prior + noise).

Provides two concrete implementations:
- ``BayesianInferenceProblem``: general inference (any DistributionProtocol prior)
- ``GaussianInferenceProblem``: Gaussian inference (adds prior_mean/prior_covariance)

Users can also write custom classes satisfying the protocols directly.
Designed to be movable to pyapprox/inverse/ later.
"""

from typing import Generic

from pyapprox.benchmarks.functions.algebraic.linear_gaussian_oed import (
    build_linear_obs_map,
)
from pyapprox.interface.functions.protocols import FunctionProtocol
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.probability.protocols.distribution import DistributionProtocol
from pyapprox.util.backends.protocols import Array, Backend


class BayesianInferenceProblem(Generic[Array]):
    """General Bayesian inference problem: obs_map + prior + noise.

    Satisfies BayesianInferenceProblemProtocol. No sampling — consumers
    choose MC/Sobol/quadrature.

    Parameters
    ----------
    obs_map : FunctionProtocol[Array]
        Observation model mapping parameters to observations.
    prior : DistributionProtocol[Array]
        Prior distribution (any distribution with bkd/nvars/rvs/logpdf).
    noise_variances : Array
        Noise variances. Shape: (nobs,). nobs derived from this.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        obs_map: FunctionProtocol[Array],
        prior: DistributionProtocol[Array],
        noise_variances: Array,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._obs_map = obs_map
        self._prior = prior
        self._noise_variances = noise_variances

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def obs_map(self) -> FunctionProtocol[Array]:
        """Get the observation map."""
        return self._obs_map

    def prior(self) -> DistributionProtocol[Array]:
        """Get the prior distribution."""
        return self._prior

    def noise_variances(self) -> Array:
        """Get noise variances. Shape: (nobs,)."""
        return self._noise_variances

    def nobs(self) -> int:
        """Number of observations (derived from noise_variances)."""
        return int(self._noise_variances.shape[0])

    def nparams(self) -> int:
        """Number of model parameters."""
        return self._obs_map.nvars()


class GaussianInferenceProblem(BayesianInferenceProblem[Array]):
    """Gaussian Bayesian inference problem.

    Satisfies GaussianInferenceProblemProtocol. Extends
    BayesianInferenceProblem with prior_mean and prior_covariance
    for conjugate Gaussian analytics.

    Parameters
    ----------
    obs_map : FunctionProtocol[Array]
        Observation model mapping parameters to observations.
    prior : DenseCholeskyMultivariateGaussian[Array]
        Gaussian prior distribution.
    noise_variances : Array
        Noise variances. Shape: (nobs,). nobs derived from this.
    bkd : Backend[Array]
        Computational backend.
    prior_mean : Array
        Prior mean. Shape: (nparams, 1).
    prior_covariance : Array
        Prior covariance. Shape: (nparams, nparams).
    """

    def __init__(
        self,
        obs_map: FunctionProtocol[Array],
        prior: DenseCholeskyMultivariateGaussian[Array],
        noise_variances: Array,
        bkd: Backend[Array],
        prior_mean: Array,
        prior_covariance: Array,
    ) -> None:
        super().__init__(obs_map, prior, noise_variances, bkd)
        self._prior_mean = prior_mean
        self._prior_covariance = prior_covariance

    def prior(self) -> DenseCholeskyMultivariateGaussian[Array]:
        """Get the Gaussian prior distribution."""
        return self._prior  # type: ignore[return-value]

    def prior_mean(self) -> Array:
        """Get prior mean. Shape: (nparams, 1)."""
        return self._prior_mean

    def prior_covariance(self) -> Array:
        """Get prior covariance. Shape: (nparams, nparams)."""
        return self._prior_covariance


def build_linear_gaussian_inference_problem(
    nobs: int,
    degree: int,
    noise_std: float,
    prior_std: float,
    bkd: Backend[Array],
    min_degree: int = 0,
) -> GaussianInferenceProblem[Array]:
    """Build a GaussianInferenceProblem for linear Gaussian regression.

    Constructs a polynomial regression observation model with isotropic
    Gaussian prior and noise.

    Parameters
    ----------
    nobs : int
        Number of observation locations (equally spaced in [-1, 1]).
    degree : int
        Maximum polynomial degree.
    noise_std : float
        Standard deviation of observation noise.
    prior_std : float
        Standard deviation of prior on coefficients.
    bkd : Backend[Array]
        Computational backend.
    min_degree : int
        Minimum polynomial degree (default 0).

    Returns
    -------
    problem : GaussianInferenceProblem
        Configured inference problem.
    """
    obs_locations = bkd.linspace(-1.0, 1.0, nobs)
    obs_map = build_linear_obs_map(obs_locations, min_degree, degree, bkd)
    nparams = obs_map.nvars()

    prior_mean = bkd.zeros((nparams, 1))
    prior_covariance = bkd.eye(nparams) * prior_std**2
    prior = DenseCholeskyMultivariateGaussian(
        prior_mean, prior_covariance, bkd
    )

    noise_variances = bkd.full((nobs,), noise_std**2)

    return GaussianInferenceProblem(
        obs_map=obs_map,
        prior=prior,
        noise_variances=noise_variances,
        bkd=bkd,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
