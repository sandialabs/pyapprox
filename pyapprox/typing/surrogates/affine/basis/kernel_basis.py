"""Kernel basis functions from kernel evaluations at fixed center points.

Given a kernel K satisfying KernelProtocol and center points mu_1, ..., mu_N,
defines basis functions phi_j(y) = K(y, mu_j). General-purpose: useful for
interpolation, regression, and density estimation.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)
from pyapprox.typing.surrogates.kernels.protocols import KernelProtocol


class KernelBasis(Generic[Array]):
    """Basis functions from kernel evaluations at fixed center points.

    Basis function j is phi_j(y) = K(y, mu_j) where K is any kernel
    satisfying KernelProtocol.

    Parameters
    ----------
    kernel : KernelProtocol[Array]
        Any kernel satisfying KernelProtocol.
    centers : Array
        Center points. Shape: (nvars, ncenters). Columns are points.

    Raises
    ------
    TypeError
        If kernel does not satisfy KernelProtocol.
    ValueError
        If centers shape is incompatible with kernel.nvars().
    """

    def __init__(
        self, kernel: KernelProtocol[Array], centers: Array
    ) -> None:
        if not isinstance(kernel, KernelProtocol):
            raise TypeError(
                f"kernel must satisfy KernelProtocol, "
                f"got {type(kernel).__name__}"
            )
        if centers.ndim != 2:
            raise ValueError(
                f"centers must be 2D with shape (nvars, ncenters), "
                f"got {centers.ndim}D array with shape {centers.shape}"
            )
        if centers.shape[0] != kernel.nvars():
            raise ValueError(
                f"centers has {centers.shape[0]} rows but kernel has "
                f"nvars={kernel.nvars()}"
            )
        self._kernel = kernel
        self._centers = centers
        self._bkd = kernel.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nbasis(self) -> int:
        """Return the number of basis functions (center points)."""
        return self._centers.shape[1]

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._kernel.nvars()

    def kernel(self) -> KernelProtocol[Array]:
        """Return the underlying kernel."""
        return self._kernel

    def centers(self) -> Array:
        """Return the center points. Shape: (nvars, ncenters)."""
        return self._centers

    def hyp_list(self) -> HyperParameterList:
        """Return the kernel's hyperparameter list."""
        return self._kernel.hyp_list()

    def __call__(self, points: Array) -> Array:
        """Evaluate all basis functions at given points.

        Parameters
        ----------
        points : Array
            Evaluation points. Shape: (nvars, npts).

        Returns
        -------
        Array
            Basis values. Shape: (npts, nbasis).
            Entry [i, j] = K(points[:, i], centers[:, j]).
        """
        return self._kernel(points, self._centers)


__all__ = ["KernelBasis"]
