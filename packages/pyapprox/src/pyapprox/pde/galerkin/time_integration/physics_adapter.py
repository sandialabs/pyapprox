"""Adapter to use Galerkin physics with time integration from pyapprox.ode.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

The adapter presents a BC-NEUTRALIZED, well-posed ODE:
  mass_matrix() = FEM mass with identity rows at essential DOFs
  f(y, t)  = spatial_residual with essential rows carrying the
             boundary velocity g_dot(t) (exact zeros for
             time-invariant constraint sets)
  jacobian = spatial_jacobian with essential rows zeroed

Why not raw rows: the assembled Galerkin row at a Dirichlet DOF is not
a valid evolution equation (integration by parts dropped its boundary
flux term), and a stage solve through the dense M^{-1} would smear
that row into every interior slope. With identity mass rows and g_dot
in the f rows, every stage solve IS the constrained stage system
[M_bc] k = [F | rows d <- g_dot(t)]: stage slopes satisfy
k[d] = g_dot(t) and interior rows receive the M_id*g_dot coupling
through the cached factorization.

Constraint VALUES for the end-of-step system stay out of this layer:
the BC-enforcing time residual wrapper replaces the constrained rows
of the residual/Jacobian with R[d] = y[d] - g(t_{n+1}), J[d,:] = e_d,
so implicit methods see exactly the same Newton system as before
(interior rows are untouched by the neutralization).

The parameterized adapter tiers and the capability-selecting factory
live in ``pyapprox.pde.models.galerkin.physics_adapter`` — the models
layer owns everything that requires both a physics and a
parameterization.
"""

from typing import Generic, Optional, Tuple, Union

import numpy as np
from scipy.sparse import diags, issparse, spmatrix

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    SparseMatrixOperator,
)
from pyapprox.ode.mass_matrix import (
    DiagonalMassMatrix,
    MassMatrixProtocol,
    create_mass_matrix,
)
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from GalerkinPhysics to ODEResidualProtocol (base tier).

    Presents the BC-neutralized ODE (see module docstring):
    - f(y) = spatial_residual with essential rows carrying g_dot(t)
    - jacobian(y) = spatial_jacobian with essential rows zeroed
    - mass_matrix() = mass with identity rows at essential DOFs

    The constraint VALUES g(t) for the end-of-step system are applied
    externally by the BC-enforcing time residual wrapper.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        The Galerkin physics to adapt. Must have spatial_residual(),
        spatial_jacobian(), and constraint_set() methods.
    lumped_mass : bool, default False
        If True, use the row-sum lumped (diagonal) mass matrix instead
        of the consistent mass. Cheaper per solve, less accurate — an
        option on the one pipeline, not a separate code path.

    Examples
    --------
    >>> ode_residual = GalerkinPhysicsToODEResidualAdapter(physics)
    >>> time_stepper = BackwardEulerHVP(ode_residual)
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        lumped_mass: bool = False,
    ) -> None:
        if not isinstance(physics, GalerkinPhysicsProtocol):
            raise TypeError(
                f"physics must satisfy GalerkinPhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        self._lumped_mass = lumped_mass
        self._constraint_set = physics.constraint_set()
        # Cached: consulted on every residual evaluation.
        self._has_boundary_velocity = (
            self._constraint_set.has_time_derivatives()
        )
        # Sparse form of the BC-neutralized mass for the sparse Newton
        # matrix; None when the mass is dense.
        self._sparse_mass: Optional[spmatrix] = None
        self._mass = self._build_mass()

    def _build_mass(self) -> MassMatrixProtocol[Array]:
        """Build the BC-neutralized mass value-object.

        Lumped mass keeps its diagonality structurally
        (DiagonalMassMatrix with 1.0 at essential DOFs); consistent
        mass gets identity rows via the constraint set.
        """
        raw_mass = self._physics.mass_matrix()
        constraint_set = self._constraint_set
        if self._lumped_mass:
            diagonal = self._lumped_diagonal(raw_mass)
            if constraint_set.ndofs():
                diagonal = self._bkd.index_update(
                    diagonal, constraint_set.dofs(), 1.0
                )
            if isinstance(self._bkd, NumpyBkd):
                self._sparse_mass = diags(
                    self.bkd().to_numpy(diagonal)
                ).tocsc()
            return DiagonalMassMatrix(diagonal, self._bkd)
        bc_mass = constraint_set.apply_to_mass(raw_mass)
        if isinstance(bc_mass, spmatrix):
            self._sparse_mass = bc_mass
        return create_mass_matrix(bc_mass, self._bkd)

    def _lumped_diagonal(self, mass: Union[spmatrix, Array]) -> Array:
        """Row sums of the mass matrix as a backend array."""
        if isinstance(mass, spmatrix):
            return self._bkd.asarray(np.asarray(mass.sum(axis=1)).ravel())
        return self._bkd.sum(mass, axis=1)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def physics(self) -> GalerkinPhysicsProtocol[Array]:
        """Return the wrapped physics object."""
        return self._physics

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate the BC-neutralized residual f(y, t).

        Essential rows carry the boundary velocity g_dot(t) (exact
        zeros for time-invariant sets), so stage solves against the
        BC-neutralized mass are the constrained stage systems. When no
        analytic g_dot is available (permitted for one-step steppers —
        the wrapper replaces these rows anyway) the essential rows are
        zeroed.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            BC-neutralized residual. Shape: (nstates,)
        """
        residual = self._physics.spatial_residual(state, self._time)
        constraint_set = self._constraint_set
        if not constraint_set.ndofs():
            return residual
        if self._has_boundary_velocity:
            return self._bkd.index_update(
                residual,
                constraint_set.dofs(),
                constraint_set.values_time_derivative(self._time),
            )
        return constraint_set.zero_entries(residual)

    def jacobian(self, state: Array) -> Array:
        """Compute the state Jacobian with essential rows zeroed.

        The essential rows of f are state-independent (they carry
        g_dot(t)), so their Jacobian rows are exactly zero.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            BC-neutralized Jacobian dF/du. Shape: (nstates, nstates)
        """
        return self._constraint_set.zero_rows(
            self._physics.spatial_jacobian(state, self._time)
        )

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the FEM mass matrix as a value-object."""
        return self._mass

    def newton_jacobian(
        self, state: Array, coefficient: float
    ) -> LinearOperatorProtocol[Array]:
        """Return M - coefficient * dF/du as a linear operator.

        Sparse FEM systems get a ``SparseMatrixOperator`` whose
        ``as_matrix()`` returns the SPARSE Newton matrix, keeping
        sparsity flowing to the implicit steppers and the BC-enforcing
        wrapper (which applies constraint rows sparsely). Dense systems
        (e.g. the torch backend) fall back to the default dense
        operator.
        """
        jacobian = self.jacobian(state)
        if (
            issparse(jacobian)
            and self._sparse_mass is not None
            and isinstance(self._bkd, NumpyBkd)
        ):
            return SparseMatrixOperator(
                self._sparse_mass - coefficient * jacobian, self.bkd()
            )
        return super().newton_jacobian(state, coefficient)

    def dirichlet_dof_info(self, time: float) -> Tuple[Array, Array]:
        """Return Dirichlet DOF indices and values at given time.

        Parameters
        ----------
        time : float
            Time at which to evaluate Dirichlet BCs.

        Returns
        -------
        Tuple[Array, Array]
            dof_indices : Array
                Global DOF indices. Shape: (ndirichlet,)
            dof_values : Array
                Exact Dirichlet values. Shape: (ndirichlet,)
        """
        return self._physics.dirichlet_dof_info(time)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"physics={type(self._physics).__name__})"
        )
