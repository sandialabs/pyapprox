"""
Monte Carlo sampler for quadrature.

This module provides a Monte Carlo sampler that generates samples from
a given probability distribution with uniform quadrature weights.
"""

from typing import Generic, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.probability.protocols import DistributionProtocol


class MonteCarloSampler(Generic[Array]):
    """
    Monte Carlo sampler using a distribution's rvs() method.

    Generates random samples from a probability distribution with
    uniform quadrature weights (1/nsamples).

    Implements QuadratureSamplerProtocol.

    Parameters
    ----------
    distribution : DistributionProtocol[Array]
        The probability distribution to sample from.
        Must implement `rvs(nsamples)` returning shape (nvars, nsamples).
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for reproducibility. Note: The seed must be managed
        by the distribution itself; this parameter is stored for documentation
        but does not affect sampling unless the distribution supports seeding.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.probability.joint import IndependentJoint
    >>> from pyapprox.typing.probability.univariate import GaussianMarginal
    >>> bkd = NumpyBkd()
    >>> marginals = [GaussianMarginal(0, 1, bkd), GaussianMarginal(0, 1, bkd)]
    >>> distribution = IndependentJoint(marginals, bkd)
    >>> sampler = MonteCarloSampler(distribution, bkd)
    >>> samples, weights = sampler.sample(100)
    >>> samples.shape  # (2, 100)
    >>> weights.shape  # (100,)
    """

    def __init__(
        self,
        distribution: DistributionProtocol[Array],
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ):
        self._distribution = distribution
        self._bkd = bkd
        self._seed = seed

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables (dimension of the distribution).
        """
        return self._distribution.nvars()

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate Monte Carlo samples with uniform weights.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Random samples from the distribution. Shape: (nvars, nsamples)
        weights : Array
            Uniform quadrature weights. Shape: (nsamples,)
            Each weight is 1/nsamples.
        """
        samples = self._distribution.rvs(nsamples)
        weights = self._bkd.ones((nsamples,)) / nsamples
        return samples, weights

    def reset(self) -> None:
        """
        Reset the sampler state.

        Note: For Monte Carlo sampling, reset is a no-op unless the
        underlying distribution supports resetting its RNG state.
        """
        pass
