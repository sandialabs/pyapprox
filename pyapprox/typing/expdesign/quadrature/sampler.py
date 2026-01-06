"""
Quadrature samplers for OED expectation computation.

Provides samplers for generating quadrature points and weights used in
computing expectations over prior and data distributions.
"""

from typing import Generic, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class QuadratureSampler(ABC, Generic[Array]):
    """
    Abstract base class for quadrature samplers.

    Samplers generate samples and weights for numerical integration.

    Parameters
    ----------
    nvars : int
        Number of random variables to sample.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
    ):
        self._bkd = bkd
        self._nvars = nvars
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars(self) -> int:
        """Number of random variables."""
        return self._nvars

    def reset(self) -> None:
        """Reset the sampler state for reproducibility."""
        self._rng = np.random.default_rng(self._seed)

    @abstractmethod
    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate quadrature samples and weights.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, nsamples)
        weights : Array
            Quadrature weights. Shape: (nsamples,)
        """
        raise NotImplementedError


class MonteCarloSampler(QuadratureSampler[Array]):
    """
    Monte Carlo sampler with uniform weights.

    Generates random samples from the standard normal distribution
    (or unit hypercube if uniform=True) with uniform quadrature weights.

    Parameters
    ----------
    nvars : int
        Number of random variables.
    bkd : Backend[Array]
        Computational backend.
    seed : int, optional
        Random seed for reproducibility.
    uniform : bool, optional
        If True, sample from uniform [0, 1]. Otherwise, standard normal.
        Default is False (standard normal).
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        seed: Optional[int] = None,
        uniform: bool = False,
    ):
        super().__init__(nvars, bkd, seed)
        self._uniform = uniform

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate Monte Carlo samples with uniform weights."""
        if self._uniform:
            samples_np = self._rng.uniform(0, 1, (self._nvars, nsamples))
        else:
            samples_np = self._rng.standard_normal((self._nvars, nsamples))

        samples = self._bkd.asarray(samples_np)
        weights = self._bkd.ones((nsamples,)) / nsamples

        return samples, weights


class HaltonSampler(QuadratureSampler[Array]):
    """
    Halton quasi-Monte Carlo sampler.

    Generates low-discrepancy samples using the Halton sequence,
    which provides better coverage than random sampling.

    Parameters
    ----------
    nvars : int
        Number of random variables.
    bkd : Backend[Array]
        Computational backend.
    start_index : int, optional
        Starting index in the Halton sequence. Default is 0.
    transform_to_normal : bool, optional
        If True, transform uniform samples to standard normal via inverse CDF.
        Default is True.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        start_index: int = 0,
        transform_to_normal: bool = True,
    ):
        super().__init__(nvars, bkd, seed=None)
        self._start_index = start_index
        self._current_index = start_index
        self._transform = transform_to_normal
        self._primes = self._get_first_n_primes(nvars)

    def reset(self) -> None:
        """Reset to starting index."""
        self._current_index = self._start_index

    @staticmethod
    def _get_first_n_primes(n: int) -> np.ndarray:
        """Get the first n prime numbers."""
        if n == 0:
            return np.array([], dtype=int)
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return np.array(primes, dtype=int)

    def _halton_sequence(self, nsamples: int) -> np.ndarray:
        """Generate Halton sequence samples in [0, 1]^nvars."""
        samples = np.zeros((self._nvars, nsamples))

        for k in range(nsamples):
            idx = self._current_index + k
            for d in range(self._nvars):
                base = self._primes[d]
                f = 1.0
                result = 0.0
                i = idx
                while i > 0:
                    f = f / base
                    result = result + f * (i % base)
                    i = i // base
                samples[d, k] = result

        self._current_index += nsamples
        return samples

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate Halton sequence samples with uniform weights."""
        samples_np = self._halton_sequence(nsamples)

        if self._transform:
            # Transform uniform to standard normal via inverse CDF
            # Clip to avoid infinities at 0 and 1
            from scipy import stats
            samples_np = np.clip(samples_np, 1e-10, 1 - 1e-10)
            samples_np = stats.norm.ppf(samples_np)

        samples = self._bkd.asarray(samples_np)
        weights = self._bkd.ones((nsamples,)) / nsamples

        return samples, weights


class GaussianQuadratureSampler(QuadratureSampler[Array]):
    """
    Tensor product Gaussian quadrature sampler.

    Generates samples and weights using tensor product of 1D Gaussian
    quadrature rules. Suitable for low-dimensional problems.

    Parameters
    ----------
    nvars : int
        Number of random variables.
    bkd : Backend[Array]
        Computational backend.
    npoints_1d : int, optional
        Number of quadrature points per dimension. Default is 5.
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        npoints_1d: int = 5,
    ):
        super().__init__(nvars, bkd, seed=None)
        self._npoints_1d = npoints_1d
        self._samples: Optional[Array] = None
        self._weights: Optional[Array] = None
        self._setup_quadrature()

    def _setup_quadrature(self) -> None:
        """Set up tensor product Gauss-Hermite quadrature."""
        from numpy.polynomial.hermite_e import hermegauss

        # Get 1D Gauss-Hermite points and weights (for standard normal)
        points_1d, weights_1d = hermegauss(self._npoints_1d)

        # Normalize weights (hermegauss uses physicist's convention)
        weights_1d = weights_1d / np.sqrt(2 * np.pi)

        # Build tensor product
        npoints_total = self._npoints_1d ** self._nvars

        # Generate all combinations
        grids = [points_1d for _ in range(self._nvars)]
        mesh = np.meshgrid(*grids, indexing='ij')
        samples_np = np.vstack([m.flatten() for m in mesh])

        # Compute tensor product weights
        weight_grids = [weights_1d for _ in range(self._nvars)]
        weight_mesh = np.meshgrid(*weight_grids, indexing='ij')
        weights_np = np.ones(npoints_total)
        for wm in weight_mesh:
            weights_np = weights_np * wm.flatten()

        self._samples = self._bkd.asarray(samples_np)
        self._weights = self._bkd.asarray(weights_np)

    def reset(self) -> None:
        """Reset does nothing for deterministic quadrature."""
        pass

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Return Gaussian quadrature samples.

        Note: nsamples is ignored; returns all quadrature points.

        Parameters
        ----------
        nsamples : int
            Ignored. Returns all npoints_1d^nvars points.

        Returns
        -------
        samples : Array
            Quadrature points. Shape: (nvars, npoints_total)
        weights : Array
            Quadrature weights. Shape: (npoints_total,)
        """
        if self._samples is None:
            self._setup_quadrature()
        return self._samples, self._weights

    def npoints(self) -> int:
        """Return the total number of quadrature points."""
        return self._npoints_1d ** self._nvars


class OEDQuadratureSampler(Generic[Array]):
    """
    OED-specific quadrature sampler.

    Generates samples from the joint prior-data distribution needed for OED.
    The joint distribution has nvars_prior + nobs dimensions:
    - First nvars_prior dimensions: prior samples
    - Last nobs dimensions: latent noise samples

    Parameters
    ----------
    prior_sampler : QuadratureSampler[Array]
        Sampler for prior distribution.
    nobs : int
        Number of observation locations.
    bkd : Backend[Array]
        Computational backend.
    noise_sampler : QuadratureSampler[Array], optional
        Sampler for latent noise. If None, uses MonteCarloSampler.
    """

    def __init__(
        self,
        prior_sampler: QuadratureSampler[Array],
        nobs: int,
        bkd: Backend[Array],
        noise_sampler: Optional[QuadratureSampler[Array]] = None,
    ):
        self._bkd = bkd
        self._prior_sampler = prior_sampler
        self._nobs = nobs

        if noise_sampler is None:
            # Default to MC for noise (standard normal)
            noise_sampler = MonteCarloSampler(nobs, bkd, seed=42)
        self._noise_sampler = noise_sampler

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def nvars_prior(self) -> int:
        """Number of prior variables."""
        return self._prior_sampler.nvars()

    def nobs(self) -> int:
        """Number of observation locations."""
        return self._nobs

    def reset(self) -> None:
        """Reset both samplers."""
        self._prior_sampler.reset()
        self._noise_sampler.reset()

    def sample_prior(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Sample from prior distribution only.

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
        return self._prior_sampler.sample(nsamples)

    def sample_joint(
        self, nsamples: int
    ) -> Tuple[Array, Array, Array]:
        """
        Sample from joint prior-data distribution.

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
        prior_samples, prior_weights = self._prior_sampler.sample(nsamples)
        latent_samples, _ = self._noise_sampler.sample(nsamples)

        # Use prior weights (assume noise is integrated separately)
        return prior_samples, latent_samples, prior_weights
