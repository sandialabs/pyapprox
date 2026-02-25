"""Kernel density basis wrapping KernelBasis with analytical mass matrix."""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter.hyperparameter_list import (
    HyperParameterList,
)
from pyapprox.surrogates.affine.basis.kernel_basis import KernelBasis
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)


class KernelDensityBasis(Generic[Array]):
    """Density basis wrapping a 1D KernelBasis with analytical mass matrix.

    For a Squared Exponential kernel K(y, mu) = exp(-0.5*(y-mu)^2/l^2),
    the mass matrix has the closed form:

        M_ij = l * sqrt(pi) * exp(-(mu_i - mu_j)^2 / (4*l^2))

    Parameters
    ----------
    kernel_basis : KernelBasis[Array]
        A KernelBasis wrapping a 1D SquaredExponentialKernel.

    Raises
    ------
    TypeError
        If the kernel_basis does not wrap a SquaredExponentialKernel.
    ValueError
        If the kernel is not 1D.
    """

    def __init__(self, kernel_basis: KernelBasis[Array]) -> None:
        kernel = kernel_basis.kernel()
        if not isinstance(kernel, SquaredExponentialKernel):
            raise TypeError(
                f"KernelDensityBasis requires SquaredExponentialKernel, "
                f"got {type(kernel).__name__}"
            )
        if kernel_basis.nvars() != 1:
            raise ValueError(
                f"KernelDensityBasis requires nvars=1, "
                f"got {kernel_basis.nvars()}"
            )
        self._kernel_basis = kernel_basis
        self._kernel = kernel
        self._bkd = kernel_basis.bkd()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nbasis(self) -> int:
        """Return the number of basis functions."""
        return self._kernel_basis.nbasis()

    def kernel_basis(self) -> KernelBasis[Array]:
        """Return the underlying KernelBasis."""
        return self._kernel_basis

    def hyp_list(self) -> HyperParameterList:
        """Return the kernel's hyperparameter list."""
        return self._kernel_basis.hyp_list()

    def _lenscale(self) -> Array:
        """Return the user-space length scale from the SE kernel."""
        return self._kernel._log_lenscale.exp_values()

    def domain(self) -> Tuple[float, float]:
        """Return the effective domain based on centers and length scale."""
        bkd = self._bkd
        centers = self._kernel_basis.centers()
        centers_1d = centers[0]
        l = self._lenscale()[0]
        margin = 4.0 * l
        c_min = bkd.min(centers_1d)
        c_max = bkd.max(centers_1d)
        return (
            float(bkd.to_numpy(c_min - margin)),
            float(bkd.to_numpy(c_max + margin)),
        )

    def evaluate(self, y_values: Array) -> Array:
        """Evaluate all basis functions at given points.

        Parameters
        ----------
        y_values : Array
            Query points. Shape: (1, npts).

        Returns
        -------
        Array
            Basis values. Shape: (nbasis, npts).
        """
        # kernel_basis expects (nvars, npts), returns (npts, nbasis)
        if y_values.ndim == 1:
            y_2d = self._bkd.reshape(y_values, (1, -1))
        else:
            y_2d = y_values
        vals = self._kernel_basis(y_2d)  # (npts, nbasis)
        return self._bkd.transpose(vals, (1, 0))  # (nbasis, npts)

    def mass_matrix(self) -> Array:
        """Compute the analytical mass matrix for SE kernel.

        M_ij = l * sqrt(pi) * exp(-(mu_i - mu_j)^2 / (4*l^2))

        Returns
        -------
        Array
            Mass matrix. Shape: (nbasis, nbasis).
        """
        bkd = self._bkd
        centers = self._kernel_basis.centers()
        mu = centers[0]  # (ncenters,)
        l = self._lenscale()[0]  # user-space length scale (scalar array)

        # Compute pairwise squared distances between centers
        mu_i = bkd.reshape(mu, (-1, 1))  # (n, 1)
        mu_j = bkd.reshape(mu, (1, -1))  # (1, n)
        diff_sq = (mu_i - mu_j) ** 2  # (n, n)

        # M_ij = l * sqrt(pi) * exp(-(mu_i - mu_j)^2 / (4*l^2))
        coeff = l * math.sqrt(math.pi)
        M = coeff * bkd.exp(-diff_sq / (4.0 * l**2))
        return M


__all__ = ["KernelDensityBasis"]
