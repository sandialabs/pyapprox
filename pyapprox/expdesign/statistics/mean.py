"""
Sample average mean statistic.

Computes the weighted mean: E[f] = sum_i w_i * f_i
"""

from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.expdesign.statistics.base import SampleStatistic


class SampleAverageMean(SampleStatistic[Array], Generic[Array]):
    """
    Compute weighted mean: E[f] = sum_i w_i * f_i.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        super().__init__(bkd)

    def jacobian_implemented(self) -> bool:
        """Jacobian is implemented."""
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute weighted mean.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Mean values. Shape: (nqoi, 1)
        """
        # values @ weights.T gives (nqoi, 1)
        return values @ weights.T

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of mean.

        d/dx E[f] = E[df/dx] = sum_i w_i * df_i/dx

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
        # einsum: sum over samples (j) weighted by weights
        # jac_values[i, j, k] * weights[j] -> result[i, k]
        return self._bkd.einsum("ijk,j->ik", jac_values, weights[0, :])

    def __repr__(self) -> str:
        return "SampleAverageMean()"
