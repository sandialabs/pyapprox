"""Essential constraints as neutralized rows of the Galerkin ODE.

The Galerkin time-integration adapter presents ``M du/dt = F(u, t)``
with its essential rows neutralized: the essential rows of ``f`` carry
the boundary velocity g_dot(t) (exact zeros for time-invariant
constraint sets) and those of the Jacobian are zeroed. How the mass
treats those rows is a strategy, injected into the adapter as a
:data:`RowTreatmentFactory`; both strategies give the same stage
equations, stage slopes k[d] = g_dot(t) and interior rows
M_II k_I = F_I - M_ID g_dot, and differ in where the coupling M_ID g_dot
sits:

- :class:`NeutralizedRows` (the default): the mass [M_II M_ID; 0 I],
  identity essential rows with their columns kept. The coupling sits in
  the mass, so when the BC-enforcing wrapper pins the end of a one-step
  residual M(y_{n+1} - y_n) - dt Phi to g(t_{n+1}), the interior rows see
  the exact boundary increment M_ID (g_{n+1} - g_n), and the total
  content 1^T M u is conserved exactly by every one-step method;
- :class:`SymmetricNeutralizedRows`: the symmetric [M_II 0; 0 I]
  (positive definite when M is), with -M_ID g_dot(t) moved into the
  interior rows of ``f``. The stepper integrates the boundary rate like
  any other forcing term (the derivative form), so content is conserved
  to the method's order, not exactly; a solver may use a symmetric
  factorization or CG.

Why not raw rows: the assembled Galerkin row at a Dirichlet DOF is not a
valid evolution equation (integration by parts dropped its boundary flux
term), and a stage solve through the dense M^{-1} would smear that row
into every interior slope.

Constraint values for the end-of-step system stay out of this layer: a
BC-enforcing time residual wrapper replaces the constrained rows of the
step residual and Jacobian with R[d] = y[d] - g(t_{n+1}), J[d, :] = e_d.
"""

from typing import Callable, Generic, Optional, Protocol, TypeVar, Union

import numpy as np
from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import DiagonalMassMatrix, create_mass_matrix
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrix,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    RowReplacedOperator,
    RowSetProtocol,
    SymmetricEliminatedOperator,
)

F = TypeVar("F", bound=FormatProtocol)


class RowTreatmentProtocol(Protocol[Array]):
    """How the Galerkin ODE treats its essential rows: the mass in both
    forms (the matrix-era value object and the stack operator), and the
    edits to ``f`` and its Jacobian, which must agree with that mass."""

    def mass(self) -> ConstrainedMassMatrix[Array]:
        """The neutralized mass value object, the raw mass alongside."""
        ...

    def stack_mass(
        self,
        assembled: ActionOperatorProtocol[Array, F],
        rows: RowSetProtocol[Array],
    ) -> ActionOperatorProtocol[Array, F]:
        """The same neutralized mass from the assembled stack mass."""
        ...

    def residual(self, residual: Array, time: float) -> Array:
        """``f``: the spatial residual with its essential rows treated."""
        ...

    def jacobian(self, jacobian: Array) -> Array:
        """The Jacobian of :meth:`residual`."""
        ...


RowTreatmentFactory = Callable[
    [ConstraintSetProtocol[Array], Backend[Array], bool, Union[spmatrix, Array]],
    RowTreatmentProtocol[Array],
]
"""Builds a row treatment from the constraint set, backend, whether the
mass is lumped, and the raw consistent mass; each strategy's class is one."""


class NeutralizedRows(Generic[Array]):
    """Neutralizes the essential rows, keeping the mass's coupling columns
    (the default strategy; see module).

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
    raw_mass : spmatrix or Array
        The system's consistent mass, before any row or column edit.
    """

    def __init__(
        self,
        constraint_set: ConstraintSetProtocol[Array],
        bkd: Backend[Array],
        lumped_mass: bool,
        raw_mass: Union[spmatrix, Array],
    ) -> None:
        self._constraint_set = constraint_set
        self._bkd = bkd
        self._lumped_mass = lumped_mass
        self._raw_mass = raw_mass
        # Captured once: consulted on every residual evaluation.
        self._boundary_velocity: Optional[Callable[[float], Array]] = (
            constraint_set.values_derivative(1)
        )

    def mass(self) -> ConstrainedMassMatrix[Array]:
        """Build the BC-neutralized mass value-object.

        Lumped mass keeps its diagonality structurally
        (DiagonalMassMatrix with 1.0 at essential DOFs); consistent
        mass gets identity rows via the constraint set. Either way the
        mass before the essential rows were replaced is kept alongside,
        for quantities read on those rows.
        """
        if self._lumped_mass:
            return self._lumped_mass_object()
        bc_mass = self._constraint_set.apply_to_mass(self._raw_mass)
        return ConstrainedMassMatrix(
            create_mass_matrix(bc_mass, self._bkd),
            create_mass_matrix(self._raw_mass, self._bkd),
        )

    def _lumped_mass_object(self) -> ConstrainedMassMatrix[Array]:
        """The lumped mass with unit essential rows (M_ID = 0, so both
        strategies agree)."""
        constraint_set = self._constraint_set
        lumped = self._lumped_diagonal(self._raw_mass)
        diagonal = lumped
        if constraint_set.ndofs():
            diagonal = self._bkd.index_update(diagonal, constraint_set.dofs(), 1.0)
        return ConstrainedMassMatrix(
            DiagonalMassMatrix(diagonal, self._bkd),
            DiagonalMassMatrix(lumped, self._bkd),
        )

    def stack_mass(
        self,
        assembled: ActionOperatorProtocol[Array, F],
        rows: RowSetProtocol[Array],
    ) -> ActionOperatorProtocol[Array, F]:
        """The assembled mass with the essential rows replaced by identity
        rows, its columns kept."""
        return RowReplacedOperator(assembled, rows)

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


class SymmetricNeutralizedRows(NeutralizedRows[Array]):
    """Neutralizes the essential rows with the symmetric mass
    [M_II 0; 0 I] and -M_ID g_dot(t) in the interior rows of ``f`` (the
    derivative form; see module).

    The stage equations are those of :class:`NeutralizedRows`, through a
    mass that keeps M's symmetry and definiteness. Content is conserved to
    the stepper's order rather than exactly, since the interior rows see
    the stepper's quadrature of the boundary rate.

    Parameters
    ----------
    constraint_set, bkd, lumped_mass, raw_mass
        As for :class:`NeutralizedRows`.

    Raises
    ------
    TypeError
        If the mass is consistent, the constraint data vary in time, and
        no analytic g_dot is available (supply it through the boundary
        condition's ``BoundarySignal``, or lump the mass): it couples into
        every interior row.
    """

    def __init__(
        self,
        constraint_set: ConstraintSetProtocol[Array],
        bkd: Backend[Array],
        lumped_mass: bool,
        raw_mass: Union[spmatrix, Array],
    ) -> None:
        super().__init__(constraint_set, bkd, lumped_mass, raw_mass)
        # Decided once: the interior rows carry -M_ID g_dot only for a
        # consistent mass (a lumped one has M_ID = 0) and time-varying data.
        self._couples = (
            not lumped_mass
            and constraint_set.ndofs() > 0
            and not constraint_set.is_time_invariant()
        )
        if self._couples and self._boundary_velocity is None:
            raise TypeError(
                "a consistent mass with time-varying essential data needs the "
                "analytic boundary velocity g_dot, which couples into the "
                "interior rows; missing for "
                f"{constraint_set.missing_derivative_bcs(1)}. Supply it as the "
                "boundary condition's BoundarySignal time derivative, or lump "
                "the mass"
            )

    def mass(self) -> ConstrainedMassMatrix[Array]:
        """The symmetric [M_II 0; 0 I]: essential columns zeroed, then
        identity rows, with the raw mass alongside. A lumped mass is the
        same in both strategies."""
        if self._lumped_mass:
            return self._lumped_mass_object()
        constraint_set = self._constraint_set
        bc_mass = constraint_set.apply_to_mass(
            constraint_set.zero_cols(self._raw_mass)
        )
        return ConstrainedMassMatrix(
            create_mass_matrix(bc_mass, self._bkd),
            create_mass_matrix(self._raw_mass, self._bkd),
        )

    def stack_mass(
        self,
        assembled: ActionOperatorProtocol[Array, F],
        rows: RowSetProtocol[Array],
    ) -> ActionOperatorProtocol[Array, F]:
        """The assembled mass with the essential rows and columns
        eliminated, keeping its symmetry and definiteness."""
        return SymmetricEliminatedOperator(assembled, rows)

    def residual(self, residual: Array, time: float) -> Array:
        """``residual`` with -M_ID g_dot(t) added to its interior rows and
        its essential rows set to g_dot(t).

        Exact zeros on the essential rows for time-invariant sets, with no
        interior term. A lumped mass with no analytic g_dot (the one case
        permitted without it) gets zeroed essential rows.
        """
        constraint_set = self._constraint_set
        if not constraint_set.ndofs():
            return residual
        velocity = self._boundary_velocity
        if velocity is None:
            return constraint_set.zero_entries(residual)
        gdot = velocity(time)
        if self._couples:
            residual = residual - self._interior_coupling(gdot)
        return self._bkd.index_update(residual, constraint_set.dofs(), gdot)

    def _interior_coupling(self, gdot: Array) -> Array:
        """M_ID g_dot on the interior rows, zero on the essential ones: one
        application of the consistent mass to g_dot placed on the essential
        entries."""
        constraint_set = self._constraint_set
        bkd = self._bkd
        dofs = constraint_set.dofs()
        nstates = int(self._raw_mass.shape[0])
        placed = bkd.index_update(bkd.zeros((nstates,)), dofs, gdot)
        raw_mass = self._raw_mass
        if isinstance(raw_mass, spmatrix):
            coupled = bkd.asarray(raw_mass @ bkd.to_numpy(placed))
        else:
            coupled = raw_mass @ placed
        return constraint_set.zero_entries(coupled)
