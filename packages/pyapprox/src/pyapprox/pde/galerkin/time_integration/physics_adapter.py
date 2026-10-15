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

from typing import Callable, Generic, Optional, Union

import numpy as np
from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import (
    DiagonalMassMatrix,
    MassMatrixProtocol,
    create_mass_matrix,
)
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.constrained_mass import (
    ConstrainedMassMatrix,
)
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinPhysicsToODEResidualAdapter(
    DefaultNewtonJacobianMixin[Array], Generic[Array]
):
    """Adapter from a transient Galerkin system to ODEResidualProtocol
    (base tier).

    Presents the BC-neutralized ODE (see module docstring):
    - f(y) = spatial_residual with essential rows carrying g_dot(t)
    - jacobian(y) = spatial_jacobian with essential rows zeroed
    - mass_matrix() = mass with identity rows at essential DOFs

    The constraint VALUES g(t) for the end-of-step system are applied
    externally by the BC-enforcing time residual wrapper.

    Parameters
    ----------
    system : GalerkinTransientSystemProtocol
        The composed system to adapt: spatial operator, constraint set
        and mass (``compose_galerkin_system(physics, bcs)``).
    lumped_mass : bool, default False
        If True, use the row-sum lumped (diagonal) mass matrix instead
        of the consistent mass. Cheaper per solve, less accurate — an
        option on the one pipeline, not a separate code path.

    Examples
    --------
    >>> system = compose_galerkin_system(physics, bcs)
    >>> ode_residual = GalerkinPhysicsToODEResidualAdapter(system)
    >>> time_stepper = BackwardEulerHVP(ode_residual)
    """

    def __init__(
        self,
        system: GalerkinTransientSystemProtocol[Array],
        lumped_mass: bool = False,
    ) -> None:
        if not isinstance(system, GalerkinTransientSystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinTransientSystemProtocol, "
                f"got {type(system).__name__}"
            )
        self._system = system
        self._spatial_operator = system.spatial_operator()
        self._bkd = system.bkd()
        self._lumped_mass = lumped_mass
        self._constraint_set = system.constraint_set()
        # Captured once: consulted on every residual evaluation.
        self._boundary_velocity: Optional[Callable[[float], Array]] = (
            self._constraint_set.values_derivative(1)
        )
        self._mass = self._build_mass()

    def _build_mass(self) -> ConstrainedMassMatrix[Array]:
        """Build the BC-neutralized mass value-object.

        Lumped mass keeps its diagonality structurally
        (DiagonalMassMatrix with 1.0 at essential DOFs); consistent
        mass gets identity rows via the constraint set. Either way the
        mass before the essential rows were replaced is kept alongside,
        for quantities read on those rows.
        """
        raw_mass = self._system.mass_matrix()
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

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """Return the adapted system."""
        return self._system

    def __call__(self, state: Array, time: float) -> Array:
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
        time : float
            Time t.

        Returns
        -------
        Array
            BC-neutralized residual. Shape: (nstates,)
        """
        residual = self._spatial_operator.spatial_residual(state, time)
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

    def jacobian(self, state: Array, time: float) -> Array:
        """Compute the state Jacobian with essential rows zeroed.

        The essential rows of f are state-independent (they carry
        g_dot(t)), so their Jacobian rows are exactly zero.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.

        Returns
        -------
        Array
            BC-neutralized Jacobian dF/du. Shape: (nstates, nstates)
        """
        return self._constraint_set.zero_rows(
            self._spatial_operator.spatial_jacobian(state, time)
        )

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the FEM mass matrix as a value-object."""
        return self._mass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(system={self._system!r})"
