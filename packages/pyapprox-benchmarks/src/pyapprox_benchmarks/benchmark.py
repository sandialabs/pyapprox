"""Benchmark dataclasses.

These dataclasses provide concrete implementations that satisfy the benchmark
protocols. They use composition, not inheritance.
"""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


@dataclass
class BoxDomain(Generic[Array]):
    """Rectangular domain with bounds."""

    _bounds: Array
    _bkd: Backend[Array]

    def bounds(self) -> Array:
        """Return bounds of shape (nvars, 2)."""
        return self._bounds

    def nvars(self) -> int:
        """Return number of variables."""
        return int(self._bounds.shape[0])

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd
