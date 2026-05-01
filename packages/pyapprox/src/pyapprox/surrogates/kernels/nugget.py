"""Nugget kernel wrapper: adds nugget * I to any kernel."""

from __future__ import annotations

from typing import Generic

import numpy as np

from pyapprox.surrogates.kernels.protocols import (
    Kernel,
    KernelHasJacobianProtocol,
    KernelProtocol,
    NumbaScalarKernelFn,
    NumbaScalarKernelProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend
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
        if not isinstance(self._inner, KernelHasJacobianProtocol):
            raise TypeError("Inner kernel does not implement jacobian")
        return self._inner.jacobian(X1, X2)

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
