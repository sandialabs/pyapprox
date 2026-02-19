"""
Sample average mean plus standard deviation statistic.

Computes: E[f] + factor * StdDev[f]

This is a common risk measure for robust optimization.

TODO: Investigate consolidating SampleStatistic classes
(expdesign/statistics/) with RiskMeasureBase classes
(probability/risk/measures.py). They compute overlapping quantities
(mean+stdev, entropic risk, AVaR) with different APIs.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.statistics.base import SampleStatistic
from pyapprox.typing.expdesign.statistics.mean import SampleAverageMean
from pyapprox.typing.expdesign.statistics.variance import SampleAverageStdev


class SampleAverageMeanPlusStdev(SampleStatistic[Array], Generic[Array]):
    """
    Compute weighted mean plus safety_factor * standard deviation.

    E[f] + safety_factor * StdDev[f]

    Parameters
    ----------
    safety_factor : float
        Multiplier for the standard deviation term.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, safety_factor: float, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        self._mean_stat = SampleAverageMean(bkd)
        self._stdev_stat = SampleAverageStdev(bkd)
        self._safety_factor = safety_factor

    def jacobian_implemented(self) -> bool:
        """Jacobian is implemented."""
        return True

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute mean + factor * stdev.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Statistic value. Shape: (nqoi, 1)
        """
        return (
            self._mean_stat(values, weights)
            + self._safety_factor * self._stdev_stat(values, weights)
        )

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of mean + factor * stdev.

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
        return (
            self._mean_stat.jacobian(values, jac_values, weights)
            + self._safety_factor
            * self._stdev_stat.jacobian(values, jac_values, weights)
        )

    def __repr__(self) -> str:
        return f"SampleAverageMeanPlusStdev(factor={self._safety_factor})"
