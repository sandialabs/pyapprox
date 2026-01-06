"""
Sample average mean statistic.

Computes the weighted mean: E[f] = sum_i w_i * f_i
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.statistics.base import SampleStatistic


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
            Sample values. Shape: (nsamples, nqoi)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Mean values. Shape: (1, nqoi)
        """
        # values.T @ weights gives (nqoi, 1), transpose to (1, nqoi)
        return (values.T @ weights).T

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of mean.

        d/dx E[f] = E[df/dx] = sum_i w_i * df_i/dx

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
        # einsum: sum over samples (i) weighted by weights
        # jac_values[i, j, k] -> result[j, k]
        return self._bkd.einsum("ijk,i->jk", jac_values, weights[:, 0])

    def __repr__(self) -> str:
        return "SampleAverageMean()"
