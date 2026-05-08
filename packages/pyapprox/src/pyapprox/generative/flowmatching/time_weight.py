"""Time-dependent weight functions for the CFM loss."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class UniformWeight(Generic[Array]):
    """Uniform time weight: w(t) = 1.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, t: Array) -> Array:
        """Evaluate weight. t: (1, ns) -> (1, ns) of ones."""
        return self._bkd.ones_like(t)
