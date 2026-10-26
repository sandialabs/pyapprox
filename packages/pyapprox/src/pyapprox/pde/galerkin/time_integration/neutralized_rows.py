"""Essential constraints as neutralized rows of the Galerkin ODE.

The Galerkin time-integration adapter presents ``M du/dt = F(u, t)``
with its essential rows neutralized:

- the mass has identity rows at the essential DOFs;
- ``f`` carries the boundary velocity g_dot(t) in those rows (exact
  zeros for time-invariant constraint sets);
- the Jacobian has those rows zeroed.

Why not raw rows: the assembled Galerkin row at a Dirichlet DOF is not a
valid evolution equation (integration by parts dropped its boundary flux
term), and a stage solve through the dense M^{-1} would smear that row
into every interior slope. With identity mass rows and g_dot in the f
rows, every stage solve is the constrained stage system
[M_bc] k = [F | rows d <- g_dot(t)]: stage slopes satisfy k[d] = g_dot(t)
and interior rows receive the M_id g_dot coupling through the cached
factorization.

Constraint values for the end-of-step system stay out of this layer: a
BC-enforcing time residual wrapper replaces the constrained rows of the
step residual and Jacobian with R[d] = y[d] - g(t_{n+1}), J[d, :] = e_d.
"""

from typing import Callable, Generic, Optional, Union

import numpy as np
from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import DiagonalMassMatrix, create_mass_matrix
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrix,
)
from pyapprox.util.backends.protocols import Array, Backend


class NeutralizedRows(Generic[Array]):
    """Neutralizes the essential rows of a Galerkin ODE (see module).

    The interim, matrix-era form of essential-constraint handling: it
    works on the system's mass and Jacobian matrices, which are typed as
    backend arrays but are SciPy sparse for skfem physics, hence the
    sparse branch. It is deleted with the BC-enforcing wrapper; its
    successors act on stack operators and never see a matrix.

    Parameters
    ----------
    constraint_set : ConstraintSetProtocol
        The system's essential constraints.
    bkd : Backend
        Computational backend.
    lumped_mass : bool
        Whether the mass is the row-sum lumped (diagonal) one.
    """

    def __init__(
        self,
        constraint_set: ConstraintSetProtocol[Array],
        bkd: Backend[Array],
        lumped_mass: bool,
    ) -> None:
        self._constraint_set = constraint_set
        self._bkd = bkd
        self._lumped_mass = lumped_mass
        # Captured once: consulted on every residual evaluation.
        self._boundary_velocity: Optional[Callable[[float], Array]] = (
            constraint_set.values_derivative(1)
        )

    def mass(self, raw_mass: Union[spmatrix, Array]) -> ConstrainedMassMatrix[Array]:
        """Build the BC-neutralized mass value-object.

        Lumped mass keeps its diagonality structurally
        (DiagonalMassMatrix with 1.0 at essential DOFs); consistent
        mass gets identity rows via the constraint set. Either way the
        mass before the essential rows were replaced is kept alongside,
        for quantities read on those rows.
        """
        constraint_set = self._constraint_set
        if self._lumped_mass:
            lumped = self._lumped_diagonal(raw_mass)
            diagonal = lumped
            if constraint_set.ndofs():
                diagonal = self._bkd.index_update(diagonal, constraint_set.dofs(), 1.0)
            return ConstrainedMassMatrix(
                DiagonalMassMatrix(diagonal, self._bkd),
                DiagonalMassMatrix(lumped, self._bkd),
            )
        bc_mass = constraint_set.apply_to_mass(raw_mass)
        return ConstrainedMassMatrix(
            create_mass_matrix(bc_mass, self._bkd),
            create_mass_matrix(raw_mass, self._bkd),
        )

    def _lumped_diagonal(self, mass: Union[spmatrix, Array]) -> Array:
        """Row sums of the mass matrix as a backend array."""
        if isinstance(mass, spmatrix):
            return self._bkd.asarray(np.asarray(mass.sum(axis=1)).ravel())
        return self._bkd.sum(mass, axis=1)

    def residual(self, residual: Array, time: float) -> Array:
        """The essential rows of ``residual`` set to g_dot(t).

        Exact zeros for time-invariant sets. When no analytic g_dot is
        available (permitted for one-step steppers, since the wrapper
        replaces these rows anyway) the essential rows are zeroed.
        """
        constraint_set = self._constraint_set
        if not constraint_set.ndofs():
            return residual
        if self._boundary_velocity is not None:
            return self._bkd.index_update(
                residual,
                constraint_set.dofs(),
                self._boundary_velocity(time),
            )
        return constraint_set.zero_entries(residual)

    def jacobian(self, jacobian: Array) -> Array:
        """``jacobian`` with the essential rows zeroed: the essential rows
        of f are state-independent (they carry g_dot(t))."""
        return self._constraint_set.zero_rows(jacobian)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lumped_mass={self._lumped_mass})"
