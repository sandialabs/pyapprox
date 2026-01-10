"""Protocols for parameter sweeps.

Defines the interface for parameter sweep classes.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class ParameterSweeperProtocol(Protocol, Generic[Array]):
    """Protocol for parameter sweep generators.

    Parameter sweepers generate samples along random directions (sweeps)
    in the input space. This is useful for visualizing function behavior
    and understanding sensitivity to parameter perturbations.
    """

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        ...

    def nvars(self) -> int:
        """Return the number of input variables."""
        ...

    def nsamples_per_sweep(self) -> int:
        """Return the number of samples in each sweep."""
        ...

    def rvs(self, nsweeps: int) -> Array:
        """Generate parameter sweep samples.

        Parameters
        ----------
        nsweeps : int
            Number of sweeps to generate.

        Returns
        -------
        Array
            Shape (nvars, nsamples_per_sweep * nsweeps) - all sweep samples.
            Samples from each sweep are stored consecutively.
        """
        ...

    def sweep_bounds(self, rotation_vec: Array) -> Tuple[float, float]:
        """Compute bounds for a sweep along a given direction.

        Parameters
        ----------
        rotation_vec : Array
            Shape (nvars, 1) - rotation vector defining sweep direction.

        Returns
        -------
        Tuple[float, float]
            (lower, upper) bounds for the sweep in canonical coordinates.
        """
        ...

    def canonical_active_samples(self) -> Array:
        """Return the canonical (1D) sweep samples.

        Returns
        -------
        Array
            Shape (nsweeps, nsamples_per_sweep) - 1D samples for each sweep.
        """
        ...
