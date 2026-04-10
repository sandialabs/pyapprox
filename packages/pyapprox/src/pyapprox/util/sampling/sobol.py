"""
Sobol quasi-Monte Carlo sampler.

This module provides a Sobol sequence sampler using scipy.stats.qmc,
with optional transformation to arbitrary distributions via inverse CDF.
"""

from typing import Generic, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.sampling.halton import DistributionWithInvCDF


class SobolSampler(Generic[Array]):
    """
    Sobol quasi-Monte Carlo sampler.

    Generates low-discrepancy samples using the Sobol sequence via scipy.
    Provides better coverage than random sampling for numerical integration.

    Supports optional transformation to arbitrary distributions via the
    distribution's `invcdf()` method.

    Implements QuadratureSamplerProtocol.

    Parameters
    ----------
    nvars : int
        Number of random variables.
    bkd : Backend[Array]
        Computational backend.
    distribution : DistributionWithInvCDF[Array], optional
        Distribution with `invcdf()` method for transforming uniform samples.
        If provided, samples are transformed via `distribution.invcdf()`.
        If None, returns uniform [0, 1] samples (or standard normal if
        transform_to_normal=True).
    start_index : int, optional
        Starting index in the Sobol sequence. Default is 0.
        If > 0, the first `start_index` samples are skipped.
    transform_to_normal : bool, optional
        If True and no distribution is provided, transform uniform samples
        to standard normal via inverse CDF. Default is False.
        Ignored if distribution is provided.
    scramble : bool, optional
        If True, use Owen scrambling for improved uniformity. Default is True.
    seed : int, optional
        Random seed for scrambling reproducibility. Default is None.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Uniform samples in [0, 1]^3
    >>> sampler = SobolSampler(3, bkd, scramble=False)
    >>> samples, weights = sampler.sample(100)
    >>> # Transform to custom distribution
    >>> from pyapprox.probability.joint import IndependentJoint
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> marginals = [UniformMarginal(-1, 1, bkd) for _ in range(3)]
    >>> dist = IndependentJoint(marginals, bkd)
    >>> sampler = SobolSampler(3, bkd, distribution=dist)
    >>> samples, weights = sampler.sample(100)  # samples in [-1, 1]^3
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        distribution: Optional[DistributionWithInvCDF[Array]] = None,
        start_index: int = 0,
        transform_to_normal: bool = False,
        scramble: bool = True,
        seed: Optional[int] = None,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._distribution = distribution
        self._start_index = start_index
        self._transform_to_normal = transform_to_normal
        self._scramble = scramble
        self._seed = seed
        self._reset_engine()

    def _reset_engine(self) -> None:
        """Create or reset the Sobol engine."""
        self._engine = qmc.Sobol(
            d=self._nvars, scramble=self._scramble, seed=self._seed
        )
        # Skip to start_index if needed
        if self._start_index > 0:
            self._engine.fast_forward(self._start_index)
        self._current_index = self._start_index

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables.
        """
        return self._nvars

    def reset(self) -> None:
        """Reset to starting index."""
        self._reset_engine()

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate Sobol sequence samples with uniform weights.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, nsamples)
        weights : Array
            Uniform quadrature weights. Shape: (nsamples,)
            Each weight is 1/nsamples.
        """
        # TODO: why use a lazy import here?
        from scipy import stats

        # Generate samples: scipy returns (nsamples, nvars)
        samples_np = self._engine.random(n=nsamples)
        # Transpose to (nvars, nsamples) to match convention
        samples_np = samples_np.T

        self._current_index += nsamples

        # Transform samples
        if self._distribution is not None:
            # Transform via distribution's inverse CDF
            uniform_samples = self._bkd.asarray(samples_np)
            samples = self._distribution.invcdf(uniform_samples)
        elif self._transform_to_normal:
            # TODO replace this with validation that start index is setting
            # so that 0 and 1 are not part of sequence, e.g. avoid start index 0
            # which corresponds to (0,...,0)
            # remove _self._transform_to_normal for arg list and as member variable

            # Transform uniform to standard normal via inverse CDF
            # Clip to avoid infinities at 0 and 1
            samples_np = np.clip(samples_np, 1e-10, 1 - 1e-10)
            samples_np = stats.norm.ppf(samples_np)
            samples = self._bkd.asarray(samples_np)
        else:
            # Return uniform samples
            samples = self._bkd.asarray(samples_np)

        weights = self._bkd.ones((nsamples,)) / nsamples

        return samples, weights
