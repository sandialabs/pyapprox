"""
Stochastic dominance risk measures.

Utility and disutility forms of Second-order Stochastic Dominance (SSD).
"""

from typing import Generic

from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend


class UtilitySSD(SampleStatistic[Array], Generic[Array]):
    """Utility form of Second-order Stochastic Dominance.

    Computes E[max(0, eta_i - Y)] for each eta value.

    The result is convex, non-negative, and non-decreasing in eta.

    Parameters
    ----------
    eta : Array
        Threshold values. Shape: (neta,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, eta: Array, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        if eta.ndim != 1:
            raise ValueError(f"eta must be 1D, got shape {eta.shape}")
        self._eta = eta

    def __call__(self, values: Array, weights: Array) -> Array:
        """Compute E[max(0, eta_i - Y)] for each eta.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (1, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Utility SSD values. Shape: (neta, 1)
        """
        self._check_weights(values, weights)
        return self._values(values, weights)

    def _values(self, values: Array, weights: Array) -> Array:
        # values shape: (1, nsamples), use values[0] -> (nsamples,)
        # eta[:, None] - values[0][None, :] -> (neta, nsamples)
        diff = self._eta[:, None] - values[0][None, :]
        clamped = self._bkd.maximum(self._bkd.zeros((1,)), diff)
        # weighted sum: (neta, nsamples) @ (nsamples,) -> (neta,)
        result = clamped @ weights[0]
        return result[:, None]

    def __repr__(self) -> str:
        return f"UtilitySSD(neta={len(self._eta)})"


class DisutilitySSD(SampleStatistic[Array], Generic[Array]):
    """Disutility form of Second-order Stochastic Dominance.

    Computes E[max(0, Y - eta_i)] for each eta value.

    The result is convex, non-negative, and non-decreasing in eta.

    Note: This matches the legacy implementation which computes
    E[max(0, eta + Y)] (using eta + samples, not samples - eta).

    Parameters
    ----------
    eta : Array
        Threshold values. Shape: (neta,)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, eta: Array, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        if eta.ndim != 1:
            raise ValueError(f"eta must be 1D, got shape {eta.shape}")
        self._eta = eta

    def __call__(self, values: Array, weights: Array) -> Array:
        """Compute E[max(0, eta_i + Y)] for each eta.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (1, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Disutility SSD values. Shape: (neta, 1)
        """
        self._check_weights(values, weights)
        return self._values(values, weights)

    def _values(self, values: Array, weights: Array) -> Array:
        # Match legacy behavior: eta[:, None] + values[0][None, :]
        diff = self._eta[:, None] + values[0][None, :]
        clamped = self._bkd.maximum(self._bkd.zeros((1,)), diff)
        result = clamped @ weights[0]
        return result[:, None]

    def __repr__(self) -> str:
        return f"DisutilitySSD(neta={len(self._eta)})"
