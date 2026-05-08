"""
Exact (non-smoothed) Average Value at Risk (AVaR).

AVaR at level beta is the expected value of X given X >= VaR_beta.
Also known as Conditional Value at Risk (CVaR) or Expected Shortfall.

This is a non-differentiable exact computation (no smoothing), ported
from the stateful AverageValueAtRisk in probability/risk/measures.py.
"""

from typing import Generic

from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend


class ExactAVaR(SampleStatistic[Array], Generic[Array]):
    """Compute exact (non-smoothed) Average Value at Risk.

    AVaR_beta = VaR_beta + 1/(1-beta) * E[(X - VaR_beta)^+]

    This is non-differentiable, so no jacobian is provided.
    For differentiable AVaR, use ``SampleAverageSmoothedAVaR``.

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

    def _compute_single(self, vals: Array, w: Array) -> Array:
        """Compute AVaR for a single sorted QoI row.

        Parameters
        ----------
        vals : Array
            Sorted sample values. Shape: (nsamples,)
        w : Array
            Corresponding sorted weights. Shape: (nsamples,)

        Returns
        -------
        Array
            Scalar AVaR value.
        """
        weight_sum = self._bkd.sum(w)
        ecdf = self._bkd.cumsum(w) / weight_sum
        var_idx = self._bkd.to_int(self._bkd.sum(ecdf < self._beta))
        var = vals[var_idx]

        if var_idx + 1 >= len(vals):
            return var

        tail_vals = vals[var_idx + 1 :]
        tail_w = w[var_idx + 1 :]
        cvar = var + 1.0 / (1.0 - self._beta) * self._bkd.dot(
            tail_vals - var, tail_w
        )
        return cvar

    def _values(self, values: Array, weights: Array) -> Array:
        """Compute exact AVaR for each QoI.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            AVaR values. Shape: (nqoi, 1)
        """
        nqoi = values.shape[0]
        results = []
        for ii in range(nqoi):
            row = values[ii]
            w = weights[0]
            idx = self._bkd.argsort(row)
            sorted_vals = row[idx]
            sorted_w = w[idx]
            results.append(self._compute_single(sorted_vals, sorted_w))
        return self._bkd.stack(results)[:, None]

    def __repr__(self) -> str:
        return f"ExactAVaR(beta={self._beta})"
