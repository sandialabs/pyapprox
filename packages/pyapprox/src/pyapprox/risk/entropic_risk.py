"""
Sample average entropic risk statistic.

Entropic risk: (1/alpha) * log(E[exp(alpha * f)])

This is also known as the certainty equivalent for exponential utility.
"""

from typing import Generic

from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend


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
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Entropic risk values. Shape: (nqoi, 1)
        """
        # exp(alpha * values): (nqoi, nsamples)
        # exp(alpha * values) @ weights.T: (nqoi, 1)
        exp_vals = self._bkd.exp(self._alpha * values)
        return self._bkd.log(exp_vals @ weights.T) / self._alpha

    def _jacobian(self, values: Array, jac_values: Array, weights: Array) -> Array:
        """
        Compute Jacobian of entropic risk.

        For g(f(x)) = log(E[exp(alpha*f)])/alpha, chain rule gives:
        dg/dx = (1/(alpha*E[exp(alpha*f)])) * E[alpha*exp(alpha*f) * df/dx]
              = E[exp(alpha*f) * df/dx] / E[exp(alpha*f)]

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        jac_values : Array
            Jacobians at samples. Shape: (nqoi, nsamples, nvars)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        # exp(alpha * values): (nqoi, nsamples)
        exp_vals = self._bkd.exp(self._alpha * values)

        # E[exp(alpha*f)]: (nqoi, 1)
        exp_mean = exp_vals @ weights.T

        # alpha * exp(alpha*f) * jac_values: (nqoi, nsamples, nvars)
        # Weighted sum over samples: (nqoi, nvars)
        weighted_jac = self._bkd.einsum(
            "ijk,j->ik",
            self._alpha * exp_vals[..., None] * jac_values,
            weights[0, :],
        )

        # Divide by alpha * E[exp(alpha*f)]
        return weighted_jac / (self._alpha * exp_mean)

    def __repr__(self) -> str:
        return f"SampleAverageEntropicRisk(alpha={self._alpha})"
