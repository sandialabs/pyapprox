"""Helper objects for Galerkin physics classes."""

from typing import Any, Callable, Generic, Optional

import numpy as np
from numpy.typing import NDArray

try:
    from skfem import asm
    from skfem.models.poisson import mass
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )

from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


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


class PerElementField:
    """Pointwise view of a field that is constant on each element.

    Each point takes the value of the element containing it. The values
    are read through ``element_values`` at every evaluation, so a field
    reassigned after construction (a parameterization updating material
    values) is seen.

    Parameters
    ----------
    element_values : Callable
        Returns the current per-element values, shape ``(nelems,)``.
    skfem_mesh : skfem.Mesh
        The mesh whose elements index ``element_values``.
    """

    def __init__(
        self,
        element_values: Callable[[], NDArray[np.floating[Any]]],
        skfem_mesh: Any,
    ) -> None:
        self._element_values = element_values
        self._find = skfem_mesh.element_finder()

    def __call__(
        self, coords: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        values = np.asarray(self._element_values())
        if values.ndim != 1:
            raise ValueError(
                "per-quadrature-point values have no value away from the "
                "quadrature points; set one value per element "
                f"(got shape {values.shape})"
            )
        elements = self._find(*coords)
        ret: NDArray[np.floating[Any]] = values[elements]
        return ret

    def __repr__(self) -> str:
        return f"PerElementField({self._element_values!r})"
