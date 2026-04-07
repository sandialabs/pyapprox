"""
Value at Risk (VaR) — empirical quantile.

VaR at level beta is the smallest value x such that P(X <= x) >= beta.
"""

from typing import Generic

from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend


class ValueAtRisk(SampleStatistic[Array], Generic[Array]):
    """Compute empirical Value at Risk (quantile).

    VaR_beta is the smallest sample value x such that
    the cumulative weight at x is >= beta.

    Parameters
    ----------
    beta : float
        Risk level in [0, 1).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, beta: float, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        if beta < 0 or beta >= 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self._beta = beta

    def _values(self, values: Array, weights: Array) -> Array:
        """Compute empirical VaR for each QoI.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            VaR values. Shape: (nqoi, 1)
        """
        nqoi = values.shape[0]
        results = []
        for ii in range(nqoi):
            row = values[ii]
            w = weights[0]
            idx = self._bkd.argsort(row)
            sorted_vals = row[idx]
            sorted_weights = w[idx]
            weight_sum = self._bkd.sum(sorted_weights)
            ecdf = self._bkd.cumsum(sorted_weights) / weight_sum
            var_idx = self._bkd.to_int(self._bkd.sum(ecdf < self._beta))
            results.append(sorted_vals[var_idx])
        return self._bkd.stack(results)[:, None]

    def __repr__(self) -> str:
        return f"ValueAtRisk(beta={self._beta})"
