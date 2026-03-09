"""
OED quadrature sampler.

This module provides a sampler for optimal experimental design that generates
samples from the joint prior-data distribution.
"""

from typing import Generic, Optional, Tuple

from pyapprox.expdesign.protocols import QuadratureSamplerProtocol
from pyapprox.util.backends.protocols import Array, Backend

# TODO: This is specific to OED as are most other samplers except sobol and halton and should be in OED module
# Should sobol and halton quadrature (we may add other cubature rules later) be separate to local, and bayesian oed modules as they are more general types of expdesign. In legacy code sobol and halton were considered expdesign because they are often used as "good" point sets for surrgate construction, but they were primarily used for qudarture. Where should quadrature modules exist now?
# this particular file is specific to OED and so may need to exist in oed modules, are there any other samplers used only for OED?

class OEDQuadratureSampler(Generic[Array]):
    """
    OED-specific quadrature sampler.

    Generates samples from the joint prior-data distribution needed for OED.
    The joint distribution has nvars_prior + nobs dimensions:
    - First nvars_prior dimensions: prior samples
    - Last nobs dimensions: latent noise samples

    Implements OEDQuadratureSamplerProtocol.

    Parameters
    ----------
    prior_sampler : QuadratureSamplerProtocol[Array]
        Sampler for prior distribution.
    nobs : int
        Number of observation locations.
    bkd : Backend[Array]
        Computational backend.
    noise_sampler : QuadratureSamplerProtocol[Array], optional
        Sampler for latent noise. If None, uses a HaltonSampler with
        transform_to_normal=True to generate standard normal samples.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.joint import IndependentJoint
    >>> from pyapprox.probability.univariate import GaussianMarginal
    >>> bkd = NumpyBkd()
    >>> # Create prior distribution
    >>> marginals = [GaussianMarginal(0, 1, bkd) for _ in range(3)]
    >>> prior_dist = IndependentJoint(marginals, bkd)
    >>> # Create samplers
    >>> from pyapprox.expdesign.quadrature import MonteCarloSampler
    >>> prior_sampler = MonteCarloSampler(prior_dist, bkd)
    >>> oed_sampler = OEDQuadratureSampler(prior_sampler, nobs=5, bkd=bkd)
    >>> # Sample from joint distribution
    >>> prior_samples, latent_samples, weights = oed_sampler.sample_joint(100)
    """

    def __init__(
        self,
        prior_sampler: QuadratureSamplerProtocol[Array],
        nobs: int,
        bkd: Backend[Array],
        noise_sampler: Optional[QuadratureSamplerProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._prior_sampler = prior_sampler
        self._nobs = nobs

        if noise_sampler is None:
            # Default to Halton with normal transform for noise
            from pyapprox.expdesign.quadrature.halton import HaltonSampler

            noise_sampler = HaltonSampler(nobs, bkd, transform_to_normal=True, seed=42)
        self._noise_sampler = noise_sampler

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def nvars_prior(self) -> int:
        """
        Return the number of prior (parameter) variables.

        Returns
        -------
        int
            Number of prior variables.
        """
        return self._prior_sampler.nvars()

    def nobs(self) -> int:
        """
        Return the number of observation locations.

        Returns
        -------
        int
            Number of observations.
        """
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

    def sample_joint(self, nsamples: int) -> Tuple[Array, Array, Array]:
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
