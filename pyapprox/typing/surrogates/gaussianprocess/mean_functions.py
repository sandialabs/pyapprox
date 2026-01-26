"""
Mean functions for Gaussian Process regression.

This module provides mean function implementations for GP regression.
Mean functions represent the prior mean m(x) before observing data.
"""

from abc import ABC, abstractmethod
from typing import Generic, Tuple
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    HyperParameter,
    HyperParameterList
)


class MeanFunction(ABC, Generic[Array]):
    """
    Abstract base class for GP mean functions.

    A mean function defines the prior mean m(x) of the Gaussian Process
    before observing any data. After observing data, the posterior mean
    becomes: μ*(x) = m(x) + k(x, X)[K + σ²I]^{-1}(y - m(X))

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    @abstractmethod
    def __call__(self, X: Array) -> Array:
        """
        Evaluate the mean function at input locations.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Mean function values, shape (1, n_points).
        """
        ...

    @abstractmethod
    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters.

        Returns
        -------
        HyperParameterList
            List of hyperparameters for this mean function.
        """
        ...

    @abstractmethod
    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute Jacobian of mean function w.r.t. hyperparameters.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Jacobian, shape (nparams, 1, n_points).
            The gradient ∂m/∂θ_i for each hyperparameter.
            For mean functions with no parameters, returns empty array.
        """
        ...

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        Backend[Array]
            The backend instance.
        """
        return self._bkd


class ZeroMean(MeanFunction[Array]):
    """
    Zero mean function: m(x) = 0 for all x.

    This is the most common choice for GP regression, assuming
    the data has been pre-processed to have zero mean or that
    we have no prior information about the mean.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for numerical operations.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> mean = ZeroMean(bkd)
    >>> X = bkd.array(np.random.randn(2, 5))
    >>> m = mean(X)
    >>> m.shape
    (1, 5)
    >>> bkd.all_bool(m == 0.0)
    True
    """

    def __init__(self, bkd: Backend[Array]):
        super().__init__(bkd)
        # No hyperparameters
        self._hyp_list = HyperParameterList([], bkd=bkd)

    def __call__(self, X: Array) -> Array:
        """
        Evaluate zero mean function.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Zeros, shape (1, n_points).
        """
        n_points = X.shape[1]
        return self._bkd.zeros((1, n_points))

    def hyp_list(self) -> HyperParameterList:
        """
        Return empty hyperparameter list.

        Returns
        -------
        HyperParameterList
            Empty list (zero mean has no hyperparameters).
        """
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute Jacobian w.r.t. hyperparameters.

        Since ZeroMean has no hyperparameters, returns empty array.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Empty array, shape (0, 1, n_points).
        """
        n_points = X.shape[1]
        return self._bkd.zeros((0, 1, n_points))

    def __repr__(self) -> str:
        """Return string representation."""
        return "ZeroMean()"


class ConstantMean(MeanFunction[Array]):
    """
    Constant mean function: m(x) = c for all x.

    This mean function returns a constant value independent of the
    input location. The constant is a hyperparameter that can be
    optimized.

    Parameters
    ----------
    constant : float
        Initial constant value.
    constant_bounds : Tuple[float, float]
        Bounds for the constant value.
    bkd : Backend[Array]
        Backend for numerical operations.
    fixed : bool, optional
        Whether the constant is fixed (not optimized). Default is False.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> mean = ConstantMean(1.5, (-10.0, 10.0), bkd)
    >>> X = bkd.array(np.random.randn(2, 5))
    >>> m = mean(X)
    >>> m.shape
    (1, 5)
    >>> bkd.all_bool(m == 1.5)
    True
    """

    def __init__(
        self,
        constant: float,
        constant_bounds: Tuple[float, float],
        bkd: Backend[Array],
        fixed: bool = False
    ):
        super().__init__(bkd)

        # Use regular HyperParameter (not log) for constant
        self._constant = HyperParameter(
            "constant",
            1,  # Scalar parameter
            [constant],
            constant_bounds,
            bkd=self._bkd,
            fixed=fixed
        )
        self._hyp_list = HyperParameterList([self._constant])

    def __call__(self, X: Array) -> Array:
        """
        Evaluate constant mean function.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Constant values, shape (1, n_points).
        """
        n_points = X.shape[1]
        constant_value = self._constant.get_values()[0]
        return self._bkd.full((1, n_points), constant_value)

    def hyp_list(self) -> HyperParameterList:
        """
        Return hyperparameter list.

        Returns
        -------
        HyperParameterList
            List containing the constant hyperparameter.
        """
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        """
        Compute Jacobian w.r.t. hyperparameters.

        For ConstantMean, m(x) = c for all x, so ∂m/∂c = 1.

        Parameters
        ----------
        X : Array
            Input locations, shape (nvars, n_points).

        Returns
        -------
        Array
            Jacobian, shape (1, 1, n_points).
            All entries are 1.0 since ∂m/∂c = 1.
        """
        n_points = X.shape[1]
        return self._bkd.ones((1, 1, n_points))

    def __repr__(self) -> str:
        """Return string representation."""
        constant_value = self._constant.get_values()[0]
        return f"ConstantMean(constant={constant_value})"
