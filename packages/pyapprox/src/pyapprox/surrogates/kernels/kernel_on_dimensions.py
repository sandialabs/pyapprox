"""Kernel wrapper that operates on a subset of input dimensions."""

from __future__ import annotations

from typing import Generic, List

from pyapprox.surrogates.kernels.protocols import (
    Kernel,
    KernelHasHVPWrtX1Protocol,
    KernelHasJacobianProtocol,
    KernelHasParameterJacobianProtocol,
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
        self._setup_derivative_methods()

    def _setup_derivative_methods(self) -> None:
        if isinstance(self._kernel, KernelHasJacobianProtocol):
            self._jac_kernel: KernelHasJacobianProtocol[Array] = (
                self._kernel
            )
            self.jacobian = self._jacobian
        if isinstance(self._kernel, KernelHasHVPWrtX1Protocol):
            self._hvp_kernel: KernelHasHVPWrtX1Protocol[Array] = (
                self._kernel
            )
            self.hvp_wrt_x1 = self._hvp_wrt_x1
        if isinstance(self._kernel, KernelHasParameterJacobianProtocol):
            self._param_jac_kernel: (
                KernelHasParameterJacobianProtocol[Array]
            ) = self._kernel
            self.jacobian_wrt_params = self._jacobian_wrt_params

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

    def _jacobian(self, X1: Array, X2: Array) -> Array:
        """Jacobian w.r.t. X1, zero-padded to total_nvars."""
        inner_jac = self._jac_kernel.jacobian(
            self._extract(X1), self._extract(X2)
        )
        return self._zero_pad_last_axis(inner_jac)

    def _hvp_wrt_x1(
        self, X1: Array, X2: Array, direction: Array
    ) -> Array:
        """HVP w.r.t. X1, extracting direction and zero-padding."""
        inner_dir = direction[self._dims]
        inner_hvp = self._hvp_kernel.hvp_wrt_x1(
            self._extract(X1), self._extract(X2), inner_dir
        )
        return self._zero_pad_last_axis(inner_hvp)

    def _jacobian_wrt_params(self, X1: Array) -> Array:
        """Jacobian w.r.t. hyperparameters — delegates directly."""
        return self._param_jac_kernel.jacobian_wrt_params(
            self._extract(X1)
        )
