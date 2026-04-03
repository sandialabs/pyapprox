"""
Protocols for quadrature samplers in OED.

OED requires computing expectations over:
1. Prior distribution (inner loop for evidence)
2. Joint prior-data distribution (outer loop for EIG)

These protocols define how to generate samples and weights for quadrature.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class OEDQuadratureSamplerProtocol(Protocol, Generic[Array]):
    """
    Protocol for OED-specific quadrature sampling.

    Extends QuadratureSamplerProtocol with methods for generating
    samples from the joint prior-data distribution needed for OED.

    The joint distribution has nvars_prior + nobs dimensions:
    - First nvars_prior dimensions: prior samples (theta)
    - Last nobs dimensions: latent noise samples (for reparameterization)

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars_prior()
        Number of prior variables.
    nobs()
        Number of observation locations.
    sample_prior(nsamples)
        Sample from prior only.
    sample_joint(nsamples)
        Sample from joint prior-data distribution.
    reset()
        Reset the sampler state.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars_prior(self) -> int:
        """
        Number of prior (parameter) variables.

        Returns
        -------
        int
            Dimension of prior.
        """
        ...

    def nobs(self) -> int:
        """
        Number of observation locations.

        Returns
        -------
        int
            Number of observations.
        """
        ...

    def sample_prior(self, nsamples: int) -> Tuple[Array, Array]:
        """
        Generate samples from prior distribution only.

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
        ...

    def sample_joint(self, nsamples: int) -> Tuple[Array, Array, Array]:
        """
        Generate samples from joint prior-data distribution.

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
        ...

    def reset(self) -> None:
        """Reset the sampler state for reproducibility."""
        ...
