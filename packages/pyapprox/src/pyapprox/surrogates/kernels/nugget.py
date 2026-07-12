"""Nugget kernel wrapper: adds nugget * I to any kernel."""

from __future__ import annotations

from typing import Generic

import numpy as np

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.surrogates.kernels.base import Kernel, KernelInputJacobian
from pyapprox.surrogates.kernels.protocols import (
    KernelProtocol,
    NumbaScalarKernelFn,
    NumbaScalarKernelProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.hyperparameter import HyperParameterList


class NuggetKernel(Kernel[Array], Generic[Array]):
    """Wraps a kernel and adds a fixed nugget to the diagonal.

    ``K_nugget(X1, X2) = K_inner(X1, X2) + nugget * I``

    If the inner kernel satisfies ``NumbaScalarKernelProtocol``, this
    wrapper also satisfies it — enabling the fused numba pivoted Cholesky
    path without materializing the kernel matrix.

    Parameters
    ----------
    kernel : Kernel[Array]
        Inner kernel to wrap.
    nugget : float
        Nugget value added to diagonal entries.
    """

    def __init__(self, kernel: KernelProtocol[Array], nugget: float) -> None:
        super().__init__(kernel.bkd())
        self._inner = kernel
        self._nugget = nugget
        self._hyp_list = kernel.hyp_list()

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def nvars(self) -> int:
        return self._inner.nvars()

    def diag(self, X1: Array) -> Array:
        return self._inner.diag(X1) + self._nugget

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        K = self._inner(X1, X2)
        if X2 is None or X2 is X1:
            K = K + self._nugget * self._bkd.eye(X1.shape[1])
        return K

    def jacobian(self, X1: Array, X2: Array) -> Array:
        inner_jac = self._inner.input_derivatives(X2).jacobian
        if inner_jac is None:
            raise NotImplementedError(
                "Inner kernel must provide an input jacobian"
            )
        # the nugget term is piecewise constant in x, so its input
        # jacobian is the inner kernel's
        return inner_jac(X1)

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """Input jacobian delegates to the inner kernel when declared."""
        if self._inner.input_derivatives(X2).jacobian is None:
            empty: Derivatives[Array] = Derivatives.none()
            return empty
        return Derivatives.first_order(
            jacobian=KernelInputJacobian(self, X2)
        )

    def numba_eval(self) -> NumbaScalarKernelFn:
        if not isinstance(self._inner, NumbaScalarKernelProtocol):
            raise TypeError(
                "Inner kernel does not satisfy NumbaScalarKernelProtocol"
            )
        from pyapprox.surrogates.kernels.matern_numba import make_nugget_eval

        inner_eval = self._inner.numba_eval()
        return make_nugget_eval(inner_eval)

    def numba_kernel_params(self) -> np.ndarray:
        if not isinstance(self._inner, NumbaScalarKernelProtocol):
            raise TypeError(
                "Inner kernel does not satisfy NumbaScalarKernelProtocol"
            )
        inner_params = self._inner.numba_kernel_params()
        return np.append(inner_params, self._nugget)
