"""Fixed sample strategy — always returns the same samples/weights.

Baseline strategy for exact evaluation within the inexact framework.
Ignores the tolerance parameter entirely.
"""

from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class FixedSampleStrategy(Generic[Array]):
    """Strategy that always returns the same pre-computed samples/weights.

    This is the baseline strategy: it ignores ``tol`` and always returns
    the full set of samples and weights. Use it when you want exact
    evaluation through the inexact framework (e.g., to verify wrapper
    neutrality).

    Parameters
    ----------
    samples : Array
        Fixed quadrature/sample points. Shape ``(n_random_vars, n_points)``.
    weights : Array
        Fixed quadrature weights. Shape ``(n_points,)``.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        samples: Array,
        weights: Array,
        bkd: Backend[Array],
    ) -> None:
        self._samples = samples
        self._weights = weights
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of random variables."""
        return int(self._samples.shape[0])

    def samples_and_weights(self, tol: float) -> Tuple[Array, Array]:
        """Return the fixed samples and weights (tol is ignored).

        Parameters
        ----------
        tol : float
            Ignored.

        Returns
        -------
        Tuple[Array, Array]
            ``(samples, weights)`` with shapes
            ``(n_random_vars, n_points)`` and ``(n_points,)``.
        """
        return self._samples, self._weights
