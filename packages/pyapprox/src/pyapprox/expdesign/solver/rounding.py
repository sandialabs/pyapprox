"""Roundings of relaxed design weights to a subset."""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class TopK(Generic[Array]):
    """Keep the ``k`` design variables with the largest weights.

    Satisfies ``RoundingProtocol``. Ties go to the lower index.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def round(self, weights: Array, k: int) -> Tuple[int, ...]:
        """Indices of the ``k`` largest weights, sorted.

        Parameters
        ----------
        weights : Array
            Relaxed design weights. Shape: (nvars, 1)
        k : int
            Number of design variables to keep, in [1, nvars].
        """
        if weights.ndim != 2 or weights.shape[1] != 1:
            raise ValueError(
                f"weights must have shape (nvars, 1), got {tuple(weights.shape)}"
            )
        nvars = weights.shape[0]
        if not 1 <= k <= nvars:
            raise ValueError(f"k must lie in [1, {nvars}], got {k}")
        values = [self._bkd.to_float(weights[ii, 0]) for ii in range(nvars)]
        # sorted is stable, so equal weights keep index order.
        largest = sorted(range(nvars), key=lambda ii: -values[ii])[:k]
        return tuple(sorted(largest))
