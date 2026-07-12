"""Kernel protocols.

A kernel is a two-argument function, so it legitimately carries BOTH
partial-derivative families, exposed through two REQUIRED accessors whose
names carry the "wrt what":

- ``param_derivatives()``: d/dtheta of ``theta -> K(X; theta)`` forms
- ``input_derivatives(X2)``: d/dx1 of ``x1 -> k(x1, X2)`` forms (the
  accessor takes the extra context and the returned bundle's fields close
  over it, so the standard bundle arities hold)

A kernel without a capability returns ``Derivatives.none()`` (the base
``Kernel`` class does this) — optional-with-getattr is forbidden. Only
in-family friend code (GP losses, prediction/acquisition gradients) reads
these accessors, each taking exactly one family. Shape conventions are
documented on the accessors (e.g. parameter jacobian ``(n, n, nparams)``
over ACTIVE parameters); the strict ``(nqoi, nvars)`` contract attaches
only to ``ObjectiveProtocol.derivatives()``.
"""

from collections.abc import Callable
from typing import Generic, Protocol, runtime_checkable

import numpy as np

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)


@runtime_checkable
class KernelProtocol(Protocol, Generic[Array]):
    """
    Protocol for kernel implementations.

    Defines the interface for kernel classes, including methods for
    evaluating the kernel, accessing derivative bundles, and handling
    hyperparameters.
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        ...

    def hyp_list(self) -> HyperParameterList[Array]:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList[Array]
            List of hyperparameters.
        """
        ...

    def nvars(self) -> int:
        """
        Return the number of input variables (dimensionality).

        Returns
        -------
        int
            Number of input dimensions.
        """
        ...

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix.
        """
        ...

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        """
        Compute the kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data.
        X2 : Array, optional
            Input data. If None, the kernel matrix is computed for X1 only.

        Returns
        -------
        kernel_matrix : Array
            Kernel matrix.
        """
        ...

    def param_derivatives(self) -> Derivatives[Array]:
        """Derivatives of ``theta -> K(samples; theta)`` forms.

        Field shape conventions (kernel-specific; NOT the
        ObjectiveProtocol contract):

        - ``jacobian``: ``(samples) -> (n, n, nactive_params)``
        - ``hvp``: ``(samples, direction) -> (n, n)`` with ``direction``
          of shape ``(nactive_params, 1)``

        A kernel without analytic parameter derivatives returns
        ``Derivatives.none()``.
        """
        ...

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """Derivatives of ``x1 -> k(x1, X2)``; fields close over ``X2``.

        Field shape conventions:

        - ``jacobian``: ``(X1) -> (n1, n2, nvars)``
        - ``hvp``: ``(X1, direction) -> (n2, nvars)`` with ``X1`` a
          single sample ``(nvars, 1)`` and ``direction`` ``(nvars, 1)``

        A kernel without analytic input derivatives returns
        ``Derivatives.none()``.
        """
        ...


NumbaScalarKernelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@runtime_checkable
class NumbaScalarKernelProtocol(Protocol):
    """Protocol for kernels that provide a numba-compiled scalar evaluator.

    Kernels satisfying this protocol can be used in fused numba algorithms
    (e.g. matrix-free pivoted Cholesky) that evaluate k(x_i, x_j) inside
    a JIT-compiled loop without materializing a dense kernel matrix.
    """

    def numba_eval(self) -> NumbaScalarKernelFn:
        """Return ``@njit`` function ``f(xi, xj, params) -> float64``.

        ``xi``, ``xj`` are ``(nvars,)`` float64 arrays;
        ``params`` is ``(nparams,)`` float64.
        """
        ...

    def numba_kernel_params(self) -> np.ndarray:
        """Return kernel parameters as a 1D float64 numpy array.

        For Matern kernels this is the exponentiated length scales.
        """
        ...


@runtime_checkable
class SeparableKernelProtocol(Protocol, Generic[Array]):
    """
    Protocol for separable (product) kernels.

    A separable kernel has the form:
        k(x, y) = prod_d k_d(x_d, y_d)

    where k_d is a 1D kernel operating on dimension d. This structure
    enables efficient computation of multidimensional integrals as
    products of 1D integrals.

    Only kernels where the dimensions truly factor satisfy this protocol:
    - SeparableProductKernel: Explicitly constructed from 1D kernels
    - SquaredExponentialKernel: exp(-0.5 * sum_d ...) = prod_d exp(-0.5 * ...)

    Note: Matern 3/2 and 5/2 kernels are NOT separable because they use
    the combined Euclidean distance inside nonlinear polynomial terms.
    """

    def nvars(self) -> int:
        """Return the number of input dimensions."""
        ...

    def get_kernel_1d(self, dim: int) -> "KernelProtocol[Array]":
        """
        Get the 1D kernel for a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension index (0 to nvars-1).

        Returns
        -------
        kernel_1d : KernelProtocol[Array]
            The 1D kernel for the given dimension.
        """
        ...
