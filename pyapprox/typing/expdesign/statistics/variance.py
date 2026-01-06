"""
Sample average variance and standard deviation statistics.

Variance: Var[f] = E[(f - E[f])^2]
Standard deviation: StdDev[f] = sqrt(Var[f])
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.expdesign.statistics.base import SampleStatistic
from pyapprox.typing.expdesign.statistics.mean import SampleAverageMean


class SampleAverageVariance(SampleStatistic[Array], Generic[Array]):
    """
    Compute weighted variance: Var[f] = E[(f - E[f])^2].

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        super().__init__(bkd)
        self._mean_stat = SampleAverageMean(bkd)

    def jacobian_implemented(self) -> bool:
        """Jacobian is implemented."""
        return True

    def _diff(self, values: Array, weights: Array) -> Array:
        """
        Compute deviation from mean.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nsamples, nqoi)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Deviations. Shape: (nqoi, nsamples)
        """
        mean = self._mean_stat(values, weights).T  # (nqoi, 1)
        return (values - mean[:, 0]).T  # (nqoi, nsamples)

    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute weighted variance.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nsamples, nqoi)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Variance values. Shape: (1, nqoi)
        """
        diff = self._diff(values, weights)  # (nqoi, nsamples)
        return (diff**2 @ weights).T  # (1, nqoi)

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of variance.

        d/dx Var[f] = 2 * E[(f - E[f]) * (df/dx - dE[f]/dx)]

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
        # Compute mean jacobian: (nqoi, nvars)
        mean_jac = self._mean_stat.jacobian(values, jac_values, weights)
        # Expand for broadcasting: (1, nqoi, nvars)
        mean_jac_expanded = mean_jac[None, :, :]

        # Deviation from mean: (nqoi, nsamples)
        diff = self._diff(values, weights)

        # jac_values - mean_jac: (nsamples, nqoi, nvars)
        jac_diff = jac_values - mean_jac_expanded

        # 2 * diff * jac_diff weighted by weights
        # diff.T: (nsamples, nqoi)
        tmp = 2 * diff.T[..., None] * jac_diff  # (nsamples, nqoi, nvars)

        # Weighted sum over samples
        return self._bkd.einsum("ijk,i->jk", tmp, weights[:, 0])

    def __repr__(self) -> str:
        return "SampleAverageVariance()"


class SampleAverageStdev(SampleAverageVariance, Generic[Array]):
    """
    Compute weighted standard deviation: StdDev[f] = sqrt(Var[f]).

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
        Compute weighted standard deviation.

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nsamples, nqoi)
        weights : Array
            Quadrature weights. Shape: (nsamples, 1)

        Returns
        -------
        Array
            Standard deviation values. Shape: (1, nqoi)
        """
        return self._bkd.sqrt(super()._values(values, weights))

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of standard deviation.

        d/dx sqrt(Var) = 1/(2*sqrt(Var)) * d/dx Var

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
        # d/dx sqrt(y) = 1/(2*sqrt(y)) * dy/dx
        variance = super()._values(values, weights)  # (1, nqoi)
        variance_jac = super()._jacobian(values, jac_values, weights)  # (nqoi, nvars)

        # 1/(2*sqrt(variance)) factor: (nqoi, 1)
        factor = 1.0 / (2.0 * self._bkd.sqrt(variance.T))

        return factor * variance_jac

    def __repr__(self) -> str:
        return "SampleAverageStdev()"
