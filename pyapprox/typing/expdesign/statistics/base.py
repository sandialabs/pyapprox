"""
Base class for sample average statistics.

Sample statistics compute expectations and risk measures over samples.
"""

from abc import ABC, abstractmethod
from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class SampleStatistic(ABC, Generic[Array]):
    """
    Base class for sample average statistics.

    Sample statistics compute weighted averages of function values, such as
    mean, variance, entropic risk, or AVaR.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def jacobian_implemented(self) -> bool:
        """Check if analytical Jacobian is implemented."""
        return False

    def _check_weights(self, values: Array, weights: Array) -> None:
        """Validate weights shape matches values."""
        nsamples = values.shape[1]
        if weights.shape != (1, nsamples):
            raise ValueError(
                f"weights has shape {weights.shape}, expected (1, {nsamples})"
            )

    def _check_jac_values_shape(self, values: Array, jac_values: Array) -> None:
        """Validate jac_values shape matches values."""
        nqoi, nsamples = values.shape
        if jac_values.shape[0] != nqoi:
            raise ValueError(
                f"jac_values first dimension {jac_values.shape[0]} "
                f"must match values first dimension {nqoi}"
            )
        if jac_values.shape[1] != nsamples:
            raise ValueError(
                f"jac_values second dimension {jac_values.shape[1]} "
                f"must match values second dimension {nsamples}"
            )

    @abstractmethod
    def _values(self, values: Array, weights: Array) -> Array:
        """
        Compute the statistic (subclass implementation).

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
        raise NotImplementedError

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
        nqoi = values.shape[0]
        self._check_weights(values, weights)
        result = self._values(values, weights)
        if result.shape != (nqoi, 1):
            raise ValueError(
                f"_values returned shape {result.shape}, expected ({nqoi}, 1)"
            )
        return result

    def _jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of statistic (subclass implementation).

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement jacobian"
        )

    def jacobian(
        self, values: Array, jac_values: Array, weights: Array
    ) -> Array:
        """
        Compute Jacobian of statistic w.r.t. upstream variables via chain rule.

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
        nqoi = values.shape[0]
        nvars = jac_values.shape[2]
        self._check_weights(values, weights)
        self._check_jac_values_shape(values, jac_values)
        result = self._jacobian(values, jac_values, weights)
        if result.shape != (nqoi, nvars):
            raise ValueError(
                f"_jacobian returned shape {result.shape}, "
                f"expected ({nqoi}, {nvars})"
            )
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
