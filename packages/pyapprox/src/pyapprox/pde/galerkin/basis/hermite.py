"""Cubic Hermite (C1) finite element basis on a 1D mesh.

Wraps scikit-fem's ``ElementLineHermite`` for fourth-order problems such
as the Euler-Bernoulli beam, whose weak form needs continuous first
derivatives.
"""

from typing import Any, Callable, Generic, Optional

import numpy as np

from pyapprox.pde.galerkin.protocols.mesh import GalerkinMeshProtocol
from pyapprox.util.backends.protocols import Array, Backend

try:
    from skfem import Basis
    from skfem.element import ElementLineHermite
except ImportError:
    from pyapprox.util.optional_deps import import_optional_dependency

    import_optional_dependency(
        "skfem", feature_name="Galerkin module", extra_name="fem"
    )

# skfem's names for the two DOFs at a node.
_DOFNAMES = ("u", "u_x")


class CubicHermiteBasis(Generic[Array]):
    r"""Cubic Hermite basis on a 1D mesh: two DOFs per node.

    The DOFs at node :math:`i` are the value :math:`w(x_i)` and the slope
    :math:`w'(x_i)`, interleaved: ``[w_0, w'_0, w_1, w'_1, ...]``. The
    basis treats them as two components (0: value, 1: slope), so DOF
    ``d`` belongs to component ``d % 2``, the convention of vector bases.
    Boundary conditions therefore select DOFs as for any basis:

    - a clamp: ``DirichletBC(basis, "left", 0.0, bkd)``;
    - a pinned end (value only):
      ``DirichletBC(basis, "right", 0.0, bkd, components=(0,))``;
    - prescribed value and slope: a callable returning the rows
      ``(w, w')``, shape ``(2, npts)``.

    Parameters
    ----------
    mesh : GalerkinMeshProtocol
        A 1D mesh (e.g. ``StructuredMesh1D``).

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.pde.galerkin.mesh import StructuredMesh1D
    >>> bkd = NumpyBkd()
    >>> mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
    >>> CubicHermiteBasis(mesh).ndofs()
    22
    """

    def __init__(self, mesh: GalerkinMeshProtocol[Array]) -> None:
        if mesh.ndim() != 1:
            raise ValueError(
                f"CubicHermiteBasis needs a 1D mesh, got ndim={mesh.ndim()}"
            )
        self._mesh = mesh
        self._bkd = mesh.bkd()
        self._skfem_basis = Basis(mesh.skfem_mesh(), ElementLineHermite())
        self._dof_locs: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def mesh(self) -> GalerkinMeshProtocol[Array]:
        """Return the underlying mesh."""
        return self._mesh

    def ndofs(self) -> int:
        """Return the number of DOFs, two per node."""
        return int(self._skfem_basis.N)

    def degree(self) -> int:
        """Return the polynomial degree, 3."""
        return 3

    def ncomponents(self) -> int:
        """Return the number of DOF components: value and slope."""
        return 2

    def skfem_basis(self) -> Basis:
        """Return the underlying skfem Basis object."""
        return self._skfem_basis

    def interpolate(self, func: Callable[..., Any]) -> Array:
        """Interpolate a function from its values and slopes at the nodes.

        Parameters
        ----------
        func : Callable
            Takes coordinates of shape (1, npts) and returns the rows
            ``(w, w')``, shape (2, npts).

        Returns
        -------
        Array
            Interleaved DOFs ``[w_0, w'_0, ...]``. Shape: (ndofs,)
        """
        nodes = self._bkd.to_numpy(self._mesh.nodes())
        rows = np.asarray(func(nodes), dtype=np.float64)
        if rows.shape != (2, nodes.shape[1]):
            raise ValueError(
                "func must return the rows (w, w') with shape "
                f"(2, {nodes.shape[1]}), got {rows.shape}"
            )
        return self._bkd.asarray(rows.T.flatten())

    def evaluate(self, coeffs: Array, points: Array) -> Array:
        """Evaluate the function :math:`w` at points.

        Parameters
        ----------
        coeffs : Array
            DOF coefficients. Shape: (ndofs,)
        points : Array
            Evaluation points. Shape: (1, npts)

        Returns
        -------
        Array
            Values :math:`w(x)`. Shape: (npts,)
        """
        probes = self._skfem_basis.probes(self._bkd.to_numpy(points))
        values = probes @ self._bkd.to_numpy(coeffs)
        return self._bkd.asarray(np.asarray(values, dtype=np.float64))

    def values(self, coeffs: Array) -> Array:
        """Return the nodal values :math:`w(x_i)`. Shape: (nnodes,)."""
        return coeffs[0::2]

    def slopes(self, coeffs: Array) -> Array:
        """Return the nodal slopes :math:`w'(x_i)`. Shape: (nnodes,)."""
        return coeffs[1::2]

    def get_dofs(
        self,
        boundary_name: str,
        components: Optional[tuple[int, ...]] = None,
    ) -> Array:
        """Return DOF indices on a named boundary.

        Parameters
        ----------
        boundary_name : str
            Name of the boundary ("left" or "right").
        components : tuple of int, optional
            Restrict to these components (0: value, 1: slope). Default:
            both.

        Returns
        -------
        Array
            Sorted DOF indices.
        """
        if components is None:
            components = (0, 1)
        for comp in components:
            if comp not in (0, 1):
                raise ValueError(
                    f"component {comp} out of range for 2 components"
                )
        skip = [_DOFNAMES[k] for k in (0, 1) if k not in components]
        dofs = np.unique(
            np.asarray(
                self._skfem_basis.get_dofs(boundary_name, skip=skip)
            ).flatten()
        )
        return self._bkd.asarray(
            dofs.astype(np.int64), dtype=self._bkd.int64_dtype()
        )

    def dof_coordinates(self) -> Array:
        """Return the node coordinate of each DOF (each node twice).

        Returns
        -------
        Array
            Shape: (1, ndofs)
        """
        if self._dof_locs is None:
            self._dof_locs = self._bkd.asarray(
                np.asarray(self._skfem_basis.doflocs, dtype=np.float64)
            )
        return self._dof_locs

    def __repr__(self) -> str:
        return f"CubicHermiteBasis(ndofs={self.ndofs()})"
