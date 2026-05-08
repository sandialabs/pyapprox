"""
Protocols for quadrature samplers.

Defines the base QuadratureSamplerProtocol for generating samples and
weights. OED-specific extensions live in expdesign.protocols.quadrature.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class QuadratureSamplerProtocol(Protocol, Generic[Array]):
    """
    Protocol for generating quadrature samples and weights.

    Supports Monte Carlo, quasi-Monte Carlo (Halton/Sobol),
    and Gaussian quadrature.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of random variables.
    sample(nsamples)
        Generate samples and weights.
    reset()
        Reset the sampler state for reproducibility.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Number of random variables to sample.

        Returns
        -------
        int
            Dimension of samples.
        """
        ...

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
            For MC/QMC, weights are uniform: 1/nsamples.
        """
        ...

    def reset(self) -> None:
        """
        Reset the sampler state.

        For reproducible sequences (QMC), resets to initial state.
        For random samplers, this may reset the RNG seed.
        """
        ...
