"""
Protocols for sample average statistics.

Sample statistics compute expectations and risk measures over samples,
used for prediction OED objectives.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SampleStatisticProtocol(Protocol, Generic[Array]):
    """
    Protocol for sample average statistics (evaluate-only).

    Sample statistics compute weighted averages of function values, such as
    mean, variance, entropic risk, or AVaR. They are used for:
    - Risk measures over prediction space
    - Noise statistics over data realizations

    Methods
    -------
    bkd()
        Get the computational backend.
    jacobian_implemented()
        Whether analytical Jacobian is available.
    __call__(values, weights)
        Compute the statistic.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def jacobian_implemented(self) -> bool:
        """
        Check if analytical Jacobian is implemented.

        Returns
        -------
        bool
            True if jacobian() method is available.
        """
        ...

    def __call__(self, values: Array, weights: Array) -> Array:
        """
        Compute the sample statistic.

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
        ...


@runtime_checkable
class DifferentiableSampleStatisticProtocol(
    SampleStatisticProtocol[Array], Protocol, Generic[Array]
):
    """
    Protocol for sample statistics with Jacobian support.

    Extends ``SampleStatisticProtocol`` with a ``jacobian`` method for
    computing the chain-rule Jacobian of the statistic w.r.t. upstream
    variables.
    """

    def jacobian(self, values: Array, jac_values: Array, weights: Array) -> Array:
        """
        Compute Jacobian of statistic w.r.t. upstream variables via chain rule.

        This computes d(stat)/d(upstream) = d(stat)/d(values) @ d(values)/d(upstream)

        Parameters
        ----------
        values : Array
            Sample values. Shape: (nqoi, nsamples)
        jac_values : Array
            Jacobians of values w.r.t. upstream variables.
            Shape: (nqoi, nsamples, nvars)
        weights : Array
            Quadrature weights. Shape: (1, nsamples)

        Returns
        -------
        Array
            Jacobian of statistic. Shape: (nqoi, nvars)
        """
        ...
