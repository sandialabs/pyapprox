"""Constant kernel: k(x, x') = constant_value.

Used as a learnable output scale when composed with other kernels
via ProductKernel, e.g. ConstantKernel * Matern52Kernel.
"""

from typing import Optional, Tuple

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameterList,
    LogHyperParameter,
)


class ConstantKernel(Kernel[Array]):
    """Constant kernel: k(x, x') = constant_value for all x, x'.

    The constant_value is stored in log space and exponentiated, so it
    is always positive.  When used in a ProductKernel composition
    ``ConstantKernel(c) * base_kernel``, it acts as a learnable output
    scale: K(x, x') = c * base_kernel(x, x').

    Parameters
    ----------
    constant_value : float
        Initial value of the kernel (user space, must be > 0).
    constant_value_bounds : Tuple[float, float]
        Bounds for hyperparameter optimisation (user space).
    nvars : int
        Input dimension (needed for protocol compliance but the kernel
        value does not depend on x).
    bkd : Backend[Array]
    fixed : bool
        If True, the constant is registered but inactive during
        optimisation.
    """

    def __init__(
        self,
        constant_value: float,
        constant_value_bounds: Tuple[float, float],
        nvars: int,
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        super().__init__(bkd)
        self._nvars = nvars
        self._log_constant = LogHyperParameter(
            name="constant_value",
            nparams=1,
            user_values=[constant_value],
            user_bounds=constant_value_bounds,
            bkd=bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList(
            [self._log_constant], bkd=bkd,
        )

    def __call__(
        self, X1: Array, X2: Optional[Array] = None,
    ) -> Array:
        if X2 is None:
            X2 = X1
        c = self._log_constant.exp_values()[0]
        return c * self._bkd.ones((X1.shape[1], X2.shape[1]))

    def diag(self, X1: Array) -> Array:
        c = self._log_constant.exp_values()[0]
        return c * self._bkd.ones((X1.shape[1],))

    def jacobian(self, X1: Array, X2: Array) -> Array:
        return self._bkd.zeros(
            (X1.shape[1], X2.shape[1], X1.shape[0]),
        )

    def jacobian_wrt_params(self, samples: Array) -> Array:
        n = samples.shape[1]
        c = self._log_constant.exp_values()[0]
        jac = c * self._bkd.ones((n, n, 1))
        return jac

    def hvp_wrt_params(
        self, samples: Array, direction: Array,
    ) -> Array:
        n = samples.shape[1]
        c = self._log_constant.exp_values()[0]
        hvp = c * direction[0] * self._bkd.ones((n, n, 1))
        return hvp

    def hvp_wrt_x1(
        self, X1: Array, X2: Array, direction: Array,
    ) -> Array:
        return self._bkd.zeros(
            (X1.shape[1], X2.shape[1], X1.shape[0]),
        )

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def nvars(self) -> int:
        return self._nvars
