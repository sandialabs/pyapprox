"""Linear (dot-product) kernel: k(x, x') = variance * x @ x'.

A GP with this kernel is equivalent to Bayesian linear regression.
"""

from typing import Optional, Tuple

from pyapprox.surrogates.kernels.base import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameterList,
    LogHyperParameter,
)


class LinearKernel(Kernel[Array]):
    """Dot-product kernel: k(x, x') = signal_variance * x.T @ x'.

    Parameters
    ----------
    signal_variance : float
        Output scale of the kernel.
    signal_variance_bounds : Tuple[float, float]
        Bounds for hyperparameter optimization.
    nvars : int
        Input dimension.
    bkd : Backend[Array]
    fixed : bool
        If True, signal_variance is registered but inactive.
    """

    def __init__(
        self,
        signal_variance: float,
        signal_variance_bounds: Tuple[float, float],
        nvars: int,
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        super().__init__(bkd)
        self._nvars = nvars
        self._signal_var = LogHyperParameter(
            name="linear_signal_variance",
            nparams=1,
            user_values=signal_variance,
            user_bounds=signal_variance_bounds,
            bkd=bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._signal_var], bkd=bkd)

    def __call__(self, X1: Array, X2: Optional[Array] = None) -> Array:
        """Evaluate kernel. X1: (nvars, n1), X2: (nvars, n2). Returns (n1, n2)."""
        if X2 is None:
            X2 = X1
        sv = self._signal_var.exp_values()[0]
        return sv * (X1.T @ X2)

    def diag(self, X: Array) -> Array:
        """Diagonal of K(X, X), shape (X.shape[1],)."""
        sv = self._signal_var.exp_values()[0]
        return sv * self._bkd.einsum("ij,ij->j", X, X)

    def hyp_list(self) -> HyperParameterList[Array]:
        return self._hyp_list

    def nvars(self) -> int:
        return self._nvars
