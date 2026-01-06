"""
Sample average entropic risk statistic.

Entropic risk: (1/alpha) * log(E[exp(alpha * f)])

This is also known as the certainty equivalent for exponential utility.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.statistics.base import SampleStatistic


class SampleAverageEntropicRisk(SampleStatistic[Array], Generic[Array]):
    """
    Compute entropic risk: (1/alpha) * log(E[exp(alpha * f)]).

    Parameters
    ----------
    alpha : float
        Risk aversion parameter. Must be positive.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, alpha: float, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._alpha = alpha

    def jacobian_implemented(self) -> bool:
        """Jacobian is implemented."""
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute entropic risk.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nsamples, nqoi)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Entropic risk values. Shape: (1, nqoi)
        """
        # exp(alpha * values): (nsamples, nqoi)
        # exp(alpha * values).T @ weights: (nqoi, 1)
        exp_vals = self._bkd.exp(self._alpha * values)
        return (self._bkd.log(exp_vals.T @ weights).T / self._alpha)

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of entropic risk.

        For g(f(x)) = log(E[exp(alpha*f)])/alpha, chain rule gives:
        dg/dx = (1/(alpha*E[exp(alpha*f)])) * E[alpha*exp(alpha*f) * df/dx]
              = E[exp(alpha*f) * df/dx] / E[exp(alpha*f)]

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nsamples, nqoi)
        jac_values : Array
            Jacobians at samples. Shape: (nsamples, nqoi, nvars)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        # exp(alpha * values): (nsamples, nqoi)
        exp_vals = self._bkd.exp(self._alpha * values)

        # E[exp(alpha*f)]: (nqoi, 1)
        exp_mean = exp_vals.T @ weights

        # alpha * exp(alpha*f) * jac_values: (nsamples, nqoi, nvars)
        # Weighted sum: (nqoi, nvars)
        weighted_jac = self._bkd.einsum(
            "ijk,i->jk",
            self._alpha * exp_vals[..., None] * jac_values,
            weights[:, 0],
        )

        # Divide by alpha * E[exp(alpha*f)]
        return weighted_jac / (self._alpha * exp_mean)

    def __repr__(self) -> str:
        return f"SampleAverageEntropicRisk(alpha={self._alpha})"
