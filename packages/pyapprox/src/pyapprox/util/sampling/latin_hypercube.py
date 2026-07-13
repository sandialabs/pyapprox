"""
Latin hypercube design sampler.

This module provides a Latin hypercube sampler using scipy.stats.qmc,
with optional transformation to arbitrary distributions via inverse CDF.

Unlike the Sobol and Halton sequences, a Latin hypercube design is not
extensible: an n-point design is only a valid Latin hypercube for exactly
n points, so samples cannot be generated incrementally across multiple
calls. Consequently :meth:`LatinHypercubeSampler.sample` may only be
called once per design; call :meth:`LatinHypercubeSampler.reset` to
generate a new design. Use :class:`~pyapprox.util.sampling.sobol.SobolSampler`
or :class:`~pyapprox.util.sampling.halton.HaltonSampler` when incremental
sampling is required.
"""

from typing import Generic, Literal, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.stats import qmc

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.sampling.halton import (
    DistributionWithInvCDF,
    _validate_distribution,
)


class LatinHypercubeSampler(Generic[Array]):
    """
    Latin hypercube design (LHS) sampler.

    Generates a Latin hypercube design using scipy.stats.qmc: each of the
    nvars one-dimensional projections of the n-point design places exactly
    one point in each of the n equal-probability strata. Provides better
    coverage than random sampling for numerical integration and
    space-filling designs for surrogate construction.

    Supports optional transformation to arbitrary distributions via the
    distribution's `invcdf()` method.

    Implements QuadratureSamplerProtocol, with the restriction that
    `sample()` may only be called once per design (see Notes).

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
    transform_to_normal : bool, optional
        If True and no distribution is provided, transform uniform samples
        to standard normal via inverse CDF. Default is False.
        Ignored if distribution is provided.
    scramble : bool, optional
        If True (default), place each point uniformly at random within its
        stratum (classic LHS of McKay, Beckman and Conover 1979). If False,
        place each point at the center of its stratum (centered LHS).
    strength : {1, 2}, optional
        Strength of the design. Default is 1 (a plain Latin hypercube).
        If 2, produce an orthogonal-array-based Latin hypercube (Tang 1993)
        with improved two-dimensional projection properties. Requires
        nsamples = p**2 with p a prime number and nvars <= p + 1.
    optimization : {None, "random-cd", "lloyd"}, optional
        Post-processing to improve the design. Default is None.
        "random-cd" performs random coordinate swaps that lower the
        centered L2 discrepancy, improving low-dimensional projections.
        "lloyd" perturbs points toward an even spacing via Lloyd-Max
        relaxation (note this can break the strict one-point-per-stratum
        property).
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Notes
    -----
    A Latin hypercube design is a function of the total number of points:
    the union of two designs is not a Latin hypercube. `sample()` therefore
    raises a `RuntimeError` if called a second time before `reset()`.
    Use `SobolSampler` or `HaltonSampler` for incremental sampling.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Latin hypercube design in [0, 1]^3
    >>> sampler = LatinHypercubeSampler(3, bkd, seed=42)
    >>> samples, weights = sampler.sample(100)
    >>> # Discrepancy-optimized design
    >>> sampler = LatinHypercubeSampler(3, bkd, optimization="random-cd")
    >>> samples, weights = sampler.sample(100)
    >>> # Transform to custom distribution
    >>> from pyapprox.probability.joint import IndependentJoint
    >>> from pyapprox.probability.univariate import UniformMarginal
    >>> marginals = [UniformMarginal(-1, 1, bkd) for _ in range(3)]
    >>> dist = IndependentJoint(marginals, bkd)
    >>> sampler = LatinHypercubeSampler(3, bkd, distribution=dist)
    >>> samples, weights = sampler.sample(100)  # samples in [-1, 1]^3
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        distribution: Optional[DistributionWithInvCDF[Array]] = None,
        transform_to_normal: bool = False,
        scramble: bool = True,
        strength: Literal[1, 2] = 1,
        optimization: Optional[Literal["random-cd", "lloyd"]] = None,
        seed: Optional[int] = None,
    ):
        _validate_distribution(distribution, nvars)
        self._bkd = bkd
        self._nvars = nvars
        self._distribution = distribution
        self._transform_to_normal = transform_to_normal
        self._scramble = scramble
        self._strength = strength
        self._optimization = optimization
        self._seed = seed
        self._reset_engine()

    def _reset_engine(self) -> None:
        """Create or reset the Latin hypercube engine."""
        # TODO: seed is deprecated in favor of rng in scipy>=1.15 (SPEC 7).
        # Migrate all qmc samplers (Sobol, Halton, LatinHypercube) together
        # once the scipy floor is raised.
        self._engine = qmc.LatinHypercube(
            d=self._nvars,
            scramble=self._scramble,
            strength=self._strength,
            optimization=self._optimization,
            seed=self._seed,
        )
        self._sampled = False

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
        """Reset the sampler so a new design can be generated.

        The engine is rebuilt with the same seed, so after a reset the
        sampler reproduces the same design for the same nsamples.
        """
        self._reset_engine()

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate a Latin hypercube design with uniform weights.

        May only be called once per design because Latin hypercube designs
        are not extensible; call `reset()` first to generate a new design.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate. If strength=2, must equal p**2
            for a prime p with nvars <= p + 1.

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, nsamples)
        weights : Array
            Uniform quadrature weights. Shape: (nsamples,)
            Each weight is 1/nsamples.

        Raises
        ------
        RuntimeError
            If called more than once without an intervening `reset()`.
        """
        if self._sampled:
            raise RuntimeError(
                "sample() has already been called on this design. Latin "
                "hypercube designs are not extensible, so subsequent calls "
                "would not combine into a valid design. Call reset() to "
                "generate a new design, or use SobolSampler/HaltonSampler "
                "for incremental sampling."
            )

        # Generate samples: scipy returns (nsamples, nvars)
        samples_np = self._engine.random(n=nsamples)
        # Transpose to (nvars, nsamples) to match convention
        samples_np = samples_np.T

        self._sampled = True

        # Transform samples
        if self._distribution is not None:
            # Transform via distribution's inverse CDF
            uniform_samples = self._bkd.asarray(samples_np)
            samples = self._distribution.invcdf(uniform_samples)
        elif self._transform_to_normal:
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
