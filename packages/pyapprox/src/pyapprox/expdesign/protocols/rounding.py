"""Protocol for turning relaxed design weights into a subset.

A relaxed solver returns weights in [0, 1]; a rounding chooses which
design variables a discrete design keeps. The result can be scored by a
subset objective or improved by an exchange search.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class RoundingProtocol(Protocol, Generic[Array]):
    """Choose ``k`` design variables from relaxed weights.

    Methods
    -------
    bkd()
        Get the computational backend.
    round(weights, k)
        Indices of the chosen design variables.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def round(self, weights: Array, k: int) -> Tuple[int, ...]:
        """Indices of the ``k`` chosen design variables.

        Parameters
        ----------
        weights : Array
            Relaxed design weights. Shape: (nvars, 1)
        k : int
            Number of design variables to keep.

        Returns
        -------
        Tuple[int, ...]
            ``k`` distinct indices, sorted.
        """
        ...
