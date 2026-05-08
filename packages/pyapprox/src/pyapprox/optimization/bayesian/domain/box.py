"""Box (hyperrectangle) domain for Bayesian optimization."""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


class BoxDomain(Generic[Array]):
    """Axis-aligned box domain defined by per-variable bounds.

    Parameters
    ----------
    bounds : Array
        Variable bounds, shape (nvars, 2). Each row is [lower, upper].
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bounds: Array, bkd: Backend[Array]) -> None:
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (nvars, 2), got {bounds.shape}"
            )
        self._bounds = bounds
        self._bkd = bkd

    def bounds(self) -> Array:
        """Return variable bounds, shape (nvars, 2)."""
        return self._bounds

    def nvars(self) -> int:
        """Return number of variables."""
        return int(self._bounds.shape[0])

    def bkd(self) -> Backend[Array]:
        """Return computational backend."""
        return self._bkd

    def contains(self, X: Array) -> Array:
        """Check if points are inside the box.

        Parameters
        ----------
        X : Array
            Points to check, shape (nvars, n).

        Returns
        -------
        Array
            Boolean array, shape (n,). True if point is in domain.
        """
        lb = self._bounds[:, 0:1]  # (nvars, 1)
        ub = self._bounds[:, 1:2]  # (nvars, 1)
        in_bounds = (X >= lb) & (X <= ub)
        return self._bkd.all_array(in_bounds, axis=0)
