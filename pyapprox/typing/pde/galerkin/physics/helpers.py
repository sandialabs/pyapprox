"""Helper objects for Galerkin physics classes."""

from typing import Generic, Optional

from skfem import asm
from skfem.models.poisson import mass

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.typing.pde.sparse_utils import solve_maybe_sparse


class ScalarMassAssembler(Generic[Array]):
    """Cached scalar mass matrix via skfem.models.poisson.mass.

    Owns the mass matrix and provides mass_matrix() and mass_solve().
    Only scalar physics use this (ADR, Burgers, Helmholtz). Vector
    physics (elasticity) assemble their own vector mass matrix.

    Parameters
    ----------
    basis : GalerkinBasisProtocol
        Finite element basis.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self, basis: GalerkinBasisProtocol[Array], bkd: Backend[Array]
    ) -> None:
        self._basis = basis
        self._bkd = bkd
        self._cached: Optional[Array] = None

    def mass_matrix(self) -> Array:
        """Return the scalar mass matrix M_ij = integral(phi_i * phi_j).

        Cached after first assembly.

        Returns
        -------
        sparse matrix
            Mass matrix in CSR format. Shape: (ndofs, ndofs)
        """
        if self._cached is None:
            self._cached = asm(mass, self._basis.skfem_basis())
        return self._cached

    def mass_solve(self, rhs: Array) -> Array:
        """Solve M * x = rhs for x.

        Parameters
        ----------
        rhs : Array
            Right-hand side. Shape: (ndofs,) or (ndofs, ncols)

        Returns
        -------
        Array
            Solution x = M^{-1} * rhs. Same shape as rhs.
        """
        return solve_maybe_sparse(self._bkd, self.mass_matrix(), rhs)
