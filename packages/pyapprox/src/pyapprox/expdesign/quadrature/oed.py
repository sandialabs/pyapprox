"""OED quadrature sampler.

This module provides a sampler for optimal experimental design that generates
samples from the joint prior-data distribution using a SINGLE joint sampler
over (nparams + nobs) dimensions.

Using a single joint sampler preserves low-discrepancy properties for QMC
methods (Sobol, Halton). Drawing prior and noise independently from separate
samplers destroys these properties.
"""

from typing import Callable, Generic, List, Tuple, Union

import numpy as np

from pyapprox.expdesign.protocols.oed import (
    BayesianInferenceProblemProtocol,
    GaussianInferenceProblemProtocol,
)
from pyapprox.util.protocols.sampling import QuadratureSamplerProtocol
from pyapprox.util.sampling.halton import DistributionWithInvCDF
from pyapprox.probability.joint import IndependentJoint
from pyapprox.probability.protocols import DistributionProtocol
from pyapprox.probability.protocols.distribution import MarginalProtocol
from pyapprox.probability.univariate import GaussianMarginal
from pyapprox.util.backends.protocols import Array, Backend

# Type for distributions accepted by from_problem sampler_factory
OEDDistribution = Union[
    DistributionWithInvCDF[Array], DistributionProtocol[Array]
]


def _is_diagonal(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Check if a matrix is diagonal within tolerance."""
    return bool(np.allclose(
        matrix - np.diag(np.diag(matrix)), 0.0, atol=atol
    ))


def build_oed_joint_distribution(
    problem: BayesianInferenceProblemProtocol[Array],
    bkd: Backend[Array],
) -> IndependentJoint[Array]:
    """Build the joint prior + noise distribution for OED sampling.

    Constructs an ``IndependentJoint`` distribution over
    (nparams + nobs) dimensions for use with QMC samplers (Halton, Sobol)
    that require component-wise inverse CDF transforms.

    The joint distribution has nparams + nobs dimensions:
    - First nparams dimensions: prior marginals
    - Last nobs dimensions: zero-mean Gaussian noise with variances
      from ``problem.noise_variances()``

    Supports two prior types via ``problem.prior()``:

    - ``IndependentJoint``: marginals are used directly.
    - ``DenseCholeskyMultivariateGaussian``: marginals are extracted
      from the diagonal of the covariance (must be diagonal).

    Parameters
    ----------
    problem : BayesianInferenceProblemProtocol
        Problem defining prior and noise model.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    IndependentJoint[Array]
        Joint distribution over (nparams + nobs) dimensions.

    Raises
    ------
    ValueError
        If the prior is a multivariate Gaussian with non-diagonal
        covariance.
    TypeError
        If the prior type is not supported.
    """
    prior_marginals: List[MarginalProtocol[Array]]
    if isinstance(problem, GaussianInferenceProblemProtocol):
        cov_np = bkd.to_numpy(problem.prior_covariance())
        if not _is_diagonal(cov_np):
            raise ValueError(
                "build_oed_joint_distribution requires a diagonal prior "
                "covariance (independent components). For correlated "
                "priors, construct the sampler manually using "
                "MonteCarloSampler with the prior's rvs() method, or "
                "build a custom distribution."
            )
        mean_np = bkd.to_numpy(problem.prior_mean()).ravel()
        prior_marginals = [
            GaussianMarginal(
                float(mean_np[i]), float(np.sqrt(cov_np[i, i])), bkd,
            )
            for i in range(problem.nparams())
        ]
    else:
        prior = problem.prior()
        if isinstance(prior, IndependentJoint):
            prior_marginals = list(prior.marginals())
        else:
            raise TypeError(
                f"Unsupported prior type: {type(prior).__name__}. "
                f"For non-Gaussian problems, prior() must return an "
                f"IndependentJoint."
            )

    # Build noise marginals (zero mean, variance from problem)
    noise_var_np = bkd.to_numpy(problem.noise_variances()).ravel()
    noise_marginals: List[MarginalProtocol[Array]] = [
        GaussianMarginal(0.0, float(np.sqrt(v)), bkd)
        for v in noise_var_np
    ]

    return IndependentJoint(prior_marginals + noise_marginals, bkd)


class OEDQuadratureSampler(Generic[Array]):
    """OED-specific quadrature sampler.

    Generates samples from the joint prior-data distribution needed for OED.
    The joint distribution has nparams + nobs dimensions:
    - First nparams dimensions: prior samples
    - Last nobs dimensions: latent noise samples

    Uses a single joint sampler to preserve low-discrepancy properties
    for QMC methods.

    Parameters
    ----------
    joint_sampler : QuadratureSamplerProtocol[Array]
        Sampler over (nparams + nobs) dimensions.
    nparams : int
        Number of parameter dimensions (split point).
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.expdesign.quadrature import SobolSampler
    >>> bkd = NumpyBkd()
    >>> nparams, nobs = 3, 5
    >>> joint_sampler = SobolSampler(nparams + nobs, bkd)
    >>> oed_sampler = OEDQuadratureSampler(joint_sampler, nparams, bkd)
    >>> prior_samples, latent_samples, weights = oed_sampler.sample_joint(100)
    """

    def __init__(
        self,
        joint_sampler: QuadratureSamplerProtocol[Array],
        nparams: int,
        bkd: Backend[Array],
    ) -> None:
        nobs = joint_sampler.nvars() - nparams
        if nobs <= 0:
            raise ValueError(
                f"joint_sampler.nvars()={joint_sampler.nvars()} must be > "
                f"nparams={nparams}"
            )
        self._bkd = bkd
        self._joint_sampler = joint_sampler
        self._nparams = nparams
        self._nobs = nobs

    @classmethod
    def from_problem(
        cls,
        problem: BayesianInferenceProblemProtocol[Array],
        sampler_factory: Callable[
            [IndependentJoint[Array], Backend[Array]],
            QuadratureSamplerProtocol[Array],
        ],
        bkd: Backend[Array],
    ) -> "OEDQuadratureSampler[Array]":
        """Construct from a problem with an independent prior.

        Builds the joint prior + noise distribution via
        ``build_oed_joint_distribution`` and passes it to the sampler
        factory.

        For correlated priors, construct the ``OEDQuadratureSampler``
        directly by passing a pre-built joint sampler to ``__init__``.

        Parameters
        ----------
        problem : BayesianInferenceProblemProtocol
            Problem defining nparams, nobs, prior, and noise.
        sampler_factory : callable
            Factory: (distribution, bkd) -> QuadratureSamplerProtocol.
            The distribution is an ``IndependentJoint`` with
            ``invcdf()`` support, suitable for QMC samplers.
        bkd : Backend[Array]
            Computational backend.

        Returns
        -------
        OEDQuadratureSampler
            Sampler with ndim = nparams + nobs.

        Raises
        ------
        ValueError
            If the prior covariance is not diagonal.

        Examples
        --------
        >>> from pyapprox.expdesign.quadrature import HaltonSampler
        >>> sampler = OEDQuadratureSampler.from_problem(
        ...     problem,
        ...     lambda dist, bkd: HaltonSampler(
        ...         dist.nvars(), bkd, distribution=dist,
        ...     ),
        ...     bkd,
        ... )
        """
        joint_dist = build_oed_joint_distribution(problem, bkd)
        joint_sampler = sampler_factory(joint_dist, bkd)
        return cls(joint_sampler, problem.nparams(), bkd)

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars_prior(self) -> int:
        """Return the number of prior (parameter) variables."""
        return self._nparams

    def nobs(self) -> int:
        """Return the number of observation locations."""
        return self._nobs

    def reset(self) -> None:
        """Reset the joint sampler."""
        self._joint_sampler.reset()

    def sample_prior(self, nsamples: int) -> Tuple[Array, Array]:
        """Sample from prior distribution only.

        Parameters
        ----------
        nsamples : int
            Number of samples.

        Returns
        -------
        samples : Array
            Prior samples. Shape: (nvars_prior, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        joint, weights = self._joint_sampler.sample(nsamples)
        return joint[: self._nparams, :], weights

    def sample_joint(
        self, nsamples: int
    ) -> Tuple[Array, Array, Array]:
        """Sample from joint prior-data distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples.

        Returns
        -------
        prior_samples : Array
            Prior samples. Shape: (nvars_prior, nsamples)
        latent_samples : Array
            Latent noise samples. Shape: (nobs, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        joint, weights = self._joint_sampler.sample(nsamples)
        prior_samples = joint[: self._nparams, :]
        latent_samples = joint[self._nparams :, :]
        return prior_samples, latent_samples, weights
