"""Adapter to use Galerkin physics with time integration from pyapprox.ode.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

The adapter presents a BC-NEUTRALIZED, well-posed ODE:
  mass_matrix() = FEM mass with identity rows at essential DOFs
  f(y, t)  = spatial_residual with essential rows carrying the
             boundary velocity g_dot(t) (exact zeros for
             time-invariant constraint sets)
  jacobian = spatial_jacobian with essential rows zeroed

The neutralization, and why raw rows are not used, is
:class:`~pyapprox.pde.galerkin.time_integration.neutralized_rows.NeutralizedRows`.
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

from typing import Generic

from pyapprox.ode.mass_matrix import (
    MassMatrixProtocol,
)
from pyapprox.ode.mixins.default_newton_jacobian import (
    DefaultNewtonJacobianMixin,
)
from pyapprox.ode.newton_jacobian_system import NewtonJacobianSystem
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.neutralized_rows import (
    NeutralizedRows,
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
        # Read by the parameterized subclasses' derivative constraints.
        self._constraint_set = system.constraint_set()
        self._rows = NeutralizedRows(self._constraint_set, self._bkd, lumped_mass)
        self._mass = self._rows.mass(system.mass_matrix())

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
        return self._rows.residual(
            self._spatial_operator.spatial_residual(state, time), time
        )

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
        return self._rows.jacobian(self._spatial_operator.spatial_jacobian(state, time))

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return the FEM mass matrix as a value-object."""
        return self._mass

    def implicit_system(self) -> NewtonJacobianSystem[Array]:
        """The residual-form system implicit steppers solve with: this
        adapter's own Newton matrix, in its SciPy storage."""
        return NewtonJacobianSystem(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(system={self._system!r})"
