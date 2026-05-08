"""
Mean functions for Gaussian Process regression.

This module provides mean function implementations for GP regression.
Mean functions represent the prior mean m(x) before observing data.
"""

from typing import Generic, List, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameter, HyperParameterList


@runtime_checkable
class MeanFunction(Protocol, Generic[Array]):
    """
    Protocol for GP mean functions.

    A mean function defines the prior mean m(x) of the Gaussian Process
    before observing any data. After observing data, the posterior mean
    becomes: μ*(x) = m(x) + k(x, X)[K + σ²I]^{-1}(y - m(X))

    Any class implementing ``__call__``, ``hyp_list``,
    ``jacobian_wrt_params``, and ``bkd`` satisfies this protocol.
    """

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

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the list of hyperparameters.

        Returns
        -------
        HyperParameterList[Array]
            List of hyperparameters for this mean function.
        """
        ...

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
        ...


class ZeroMean(Generic[Array]):
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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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
        self._bkd = bkd
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

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return empty hyperparameter list.

        Returns
        -------
        HyperParameterList[Array]
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

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        Backend[Array]
            The backend instance.
        """
        return self._bkd

    def __repr__(self) -> str:
        """Return string representation."""
        return "ZeroMean()"


class ConstantMean(Generic[Array]):
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
    >>> from pyapprox.util.backends.numpy import NumpyBkd
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
        fixed: bool = False,
    ):
        self._bkd = bkd

        # Use regular HyperParameter (not log) for constant
        self._constant = HyperParameter(
            "constant",
            1,  # Scalar parameter
            [constant],
            constant_bounds,
            bkd=self._bkd,
            fixed=fixed,
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

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return hyperparameter list.

        Returns
        -------
        HyperParameterList[Array]
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

    def bkd(self) -> Backend[Array]:
        """
        Return the backend.

        Returns
        -------
        Backend[Array]
            The backend instance.
        """
        return self._bkd

    def __repr__(self) -> str:
        """Return string representation."""
        constant_value = self._constant.get_values()[0]
        return f"ConstantMean(constant={constant_value})"


class LinearMean(Generic[Array]):
    """Linear mean function: m(x) = w^T x + b (scalar output).

    Parameters
    ----------
    nvars : int
        Number of input variables.
    bkd : Backend[Array]
        Backend for numerical operations.
    weights_init : Optional list/array of floats
        Initial weights. Defaults to zeros.
    bias_init : float
        Initial bias. Default 0.0.
    bounds : Tuple[float, float]
        Bounds for all parameters (weights and bias).
    """

    def __init__(
        self,
        nvars: int,
        bkd: Backend[Array],
        weights_init: object = None,
        bias_init: float = 0.0,
        bounds: Tuple[float, float] = (-1e6, 1e6),
    ):
        self._bkd = bkd
        self._nvars = nvars

        if weights_init is None:
            w_vals = bkd.zeros((nvars,))
        else:
            w_vals = bkd.asarray(weights_init)

        self._weights = HyperParameter(
            "linear_weights", nvars, w_vals, bounds, bkd=bkd,
        )
        self._bias = HyperParameter(
            "linear_bias", 1, [bias_init], bounds, bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._weights, self._bias])

    def __call__(self, X: Array) -> Array:
        bkd = self._bkd
        w = self._weights.get_values()
        b = self._bias.get_values()[0]
        return bkd.reshape(bkd.dot(w, X) + b, (1, X.shape[1]))

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        """Shape (nvars+1, 1, n_points): first nvars rows are x_i, last is 1."""
        bkd = self._bkd
        n_points = X.shape[1]
        nparams = self._nvars + 1
        jac = bkd.zeros((nparams, 1, n_points))
        jac[:self._nvars, 0, :] = X
        jac[self._nvars, 0, :] = bkd.ones((n_points,))
        return jac

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return f"LinearMean(nvars={self._nvars})"


class IdentityProjection(Generic[Array]):
    """Mean function that extracts a single input dimension: m(x) = x[index, :].

    No hyperparameters.

    Parameters
    ----------
    index : int
        Which input dimension to extract.
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, index: int, bkd: Backend[Array]):
        self._index = index
        self._bkd = bkd
        self._hyp_list = HyperParameterList([], bkd=bkd)

    def __call__(self, X: Array) -> Array:
        return self._bkd.reshape(X[self._index, :], (1, X.shape[1]))

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        return self._bkd.zeros((0, 1, X.shape[1]))

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return f"IdentityProjection(index={self._index})"


class ParentPassthroughMean(Generic[Array]):
    """Mean function that extracts a parent output from an augmented input.

    m([x, f_parents...]) = augmented_input[parent_start, :]

    The parent_start index is resolved at layer-construction time.

    Parameters
    ----------
    parent_start : int
        Index in the augmented input where the target parent's output begins.
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(self, parent_start: int, bkd: Backend[Array]):
        self._parent_start = parent_start
        self._bkd = bkd
        self._hyp_list = HyperParameterList([], bkd=bkd)

    def __call__(self, X: Array) -> Array:
        return self._bkd.reshape(
            X[self._parent_start, :], (1, X.shape[1])
        )

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        return self._bkd.zeros((0, 1, X.shape[1]))

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return f"ParentPassthroughMean(parent_start={self._parent_start})"


class CompositeMean(Generic[Array]):
    """Additive composition of mean functions: m(x) = sum_i m_i(x).

    Parameters
    ----------
    components : List of MeanFunction
        Mean functions to combine additively.
    bkd : Backend[Array]
        Backend for numerical operations.
    """

    def __init__(
        self,
        components: List[MeanFunction[Array]],
        bkd: Backend[Array],
    ):
        self._components = components
        self._bkd = bkd
        all_hyps = []
        for c in components:
            all_hyps.extend(c.hyp_list().hyperparameters())
        self._hyp_list = HyperParameterList(all_hyps)

    def __call__(self, X: Array) -> Array:
        result = self._components[0](X)
        for c in self._components[1:]:
            result = result + c(X)
        return result

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def jacobian_wrt_params(self, X: Array) -> Array:
        bkd = self._bkd
        parts = [c.jacobian_wrt_params(X) for c in self._components]
        if all(p.shape[0] == 0 for p in parts):
            return bkd.zeros((0, 1, X.shape[1]))
        return bkd.concatenate(
            [p for p in parts if p.shape[0] > 0], axis=0
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __repr__(self) -> str:
        return f"CompositeMean(n_components={len(self._components)})"
