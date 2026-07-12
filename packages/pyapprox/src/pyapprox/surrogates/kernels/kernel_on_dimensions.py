"""Kernel wrapper that operates on a subset of input dimensions."""

from __future__ import annotations

from typing import Generic, List

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.surrogates.kernels.base import (
    Kernel,
    KernelInputHVP,
    KernelInputJacobian,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)


class KernelOnDimensions(Kernel[Array], Generic[Array]):
    """Kernel operating on a subset of input dimensions.

    Given an inner kernel k operating on len(dims) variables, this
    wrapper extracts X[dims, :] before delegating to the inner kernel
    while reporting nvars() = total_nvars.

    Parameters
    ----------
    kernel : Kernel[Array]
        Inner kernel. Must have nvars() == len(dims).
    dims : List[int]
        Indices of dimensions to extract.
    total_nvars : int
        Total number of input variables (reported by nvars()).
    """

    def __init__(
        self,
        kernel: Kernel[Array],
        dims: List[int],
        total_nvars: int,
    ) -> None:
        if kernel.nvars() != len(dims):
            raise ValueError(
                f"Inner kernel nvars ({kernel.nvars()}) must equal "
                f"len(dims) ({len(dims)})"
            )
        if len(dims) > total_nvars:
            raise ValueError(
                f"len(dims) ({len(dims)}) must be <= "
                f"total_nvars ({total_nvars})"
            )
        if any(d < 0 or d >= total_nvars for d in dims):
            raise ValueError(
                f"All dims must be in [0, {total_nvars}), got {dims}"
            )
        super().__init__(kernel.bkd())
        self._kernel = kernel
        self._dims = dims
        self._total_nvars = total_nvars
        # capture inner parameter capability ONCE (construction-time)
        self._inner_param_jac = kernel.param_derivatives().jacobian

    def param_derivatives(self) -> Derivatives[Array]:
        """Parameter jacobian delegates to the inner kernel when it
        declares one."""
        if self._inner_param_jac is None:
            empty: Derivatives[Array] = Derivatives.none()
            return empty
        return Derivatives.first_order(jacobian=self._jacobian_wrt_params)

    def input_derivatives(self, X2: Array) -> Derivatives[Array]:
        """Input derivatives delegate to the inner kernel on the selected
        dimensions, zero-padded to total_nvars."""
        inner = self._kernel.input_derivatives(self._extract(X2))
        if inner.jacobian is None:
            empty: Derivatives[Array] = Derivatives.none()
            return empty
        jacobian = KernelInputJacobian(self, X2)
        if inner.hvp is None:
            return Derivatives.first_order(jacobian=jacobian)
        return Derivatives(jacobian=jacobian, hvp=KernelInputHVP(self, X2))

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._kernel.hyp_list()

    def nvars(self) -> int:
        return self._total_nvars

    def dims(self) -> List[int]:
        """Return the dimension indices this kernel operates on."""
        return list(self._dims)

    def inner_kernel(self) -> Kernel[Array]:
        """Return the wrapped inner kernel."""
        return self._kernel

    def _extract(self, X: Array) -> Array:
        return X[self._dims, :]

    def diag(self, X1: Array) -> Array:
        return self._kernel.diag(self._extract(X1))

    def __call__(self, X1: Array, X2: Array | None = None) -> Array:
        X1_sub = self._extract(X1)
        X2_sub = self._extract(X2) if X2 is not None else None
        return self._kernel(X1_sub, X2_sub)

    def _zero_pad_last_axis(self, inner: Array) -> Array:
        """Embed (n1, n2, len(dims)) into (n1, n2, total_nvars) with zeros."""
        bkd = self._bkd
        n1, n2 = inner.shape[0], inner.shape[1]
        columns = []
        dim_set = {d: ii for ii, d in enumerate(self._dims)}
        for d in range(self._total_nvars):
            if d in dim_set:
                columns.append(inner[:, :, dim_set[d] : dim_set[d] + 1])
            else:
                columns.append(bkd.zeros((n1, n2, 1)))
        return bkd.concatenate(columns, axis=2)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """Jacobian w.r.t. X1, zero-padded to total_nvars."""
        inner_jac = self._kernel.input_derivatives(
            self._extract(X2)
        ).jacobian
        if inner_jac is None:
            raise NotImplementedError(
                "Inner kernel must provide an input jacobian"
            )
        return self._zero_pad_last_axis(inner_jac(self._extract(X1)))

    def hvp_wrt_x1(self, X1: Array, X2: Array, direction: Array) -> Array:
        """HVP w.r.t. X1, extracting direction and zero-padding."""
        inner_hvp = self._kernel.input_derivatives(
            self._extract(X2)
        ).hvp
        if inner_hvp is None:
            raise NotImplementedError(
                "Inner kernel must provide an input hvp"
            )
        inner_dir = direction[self._dims]
        return self._zero_pad_last_axis(
            inner_hvp(self._extract(X1), inner_dir)
        )

    def _jacobian_wrt_params(self, X1: Array) -> Array:
        """Jacobian w.r.t. hyperparameters — delegates directly."""
        inner_param_jac = self._inner_param_jac
        if inner_param_jac is None:
            raise RuntimeError(
                "_jacobian_wrt_params requires the inner kernel to "
                "declare a parameter jacobian"
            )
        return inner_param_jac(self._extract(X1))
