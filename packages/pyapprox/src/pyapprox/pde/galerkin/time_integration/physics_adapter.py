"""Adapter to use Galerkin physics with time integration from pyapprox.ode.

The time module expects ODEResidualProtocol: M * dy/dt = f(y, t)
Galerkin physics provides: M * du/dt = F(u, t)

The adapter presents a BC-NEUTRALIZED, well-posed ODE:
  mass_matrix() = FEM mass with identity rows at essential DOFs
  f(y, t)  = spatial_residual with essential rows carrying the
             boundary velocity g_dot(t) (exact zeros for
             time-invariant constraint sets)
  jacobian = spatial_jacobian with essential rows zeroed

The row treatment is an injected strategy: the default keeps the mass's
coupling columns (exact discrete conservation with the wrapper); the
symmetric one zeros them and moves -M_ID g_dot into f. Both, and why raw
rows are not used, are in
:mod:`~pyapprox.pde.galerkin.time_integration.neutralized_rows`.
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

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.mass_form_system import MassFormSystem
from pyapprox.ode.mass_matrix import (
    MassMatrixProtocol,
)
from pyapprox.ode.mass_matrix_system import MassMatrixSystem
from pyapprox.ode.protocols import ODEResidualProtocol
from pyapprox.ode.residual_derivatives import Jacobian, ODEResidualDerivatives
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.essential_pinned_rows import (
    EssentialPinnedRows,
)
from pyapprox.pde.galerkin.time_integration.jacobian_operator import (
    GalerkinJacobianOperator,
)
from pyapprox.pde.galerkin.time_integration.neutralized_rows import (
    NeutralizedRows,
    RowTreatmentFactory,
    RowTreatmentProtocol,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    FormatProtocol,
    InnerProductSpaceProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearSolverProtocol,
)

F = TypeVar("F", bound=FormatProtocol)


@runtime_checkable
class GalerkinODEResidualProtocol(ODEResidualProtocol[Array], Protocol[Array]):
    """What a Galerkin model needs of its time-integration adapter,
    whatever stack it was built on: the ODE residual, the system it
    adapts, and whether its mass is lumped."""

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """Return the adapted system."""
        ...

    def lumped_mass(self) -> bool:
        """Whether the mass is lumped (diagonal)."""
        ...


class GalerkinPhysicsToODEResidualAdapter(Generic[Array, F]):
    """Adapter from a transient Galerkin system on a stack to
    ODEResidualProtocol (base tier).

    Presents the BC-neutralized ODE (see module docstring):
    - f(y) = spatial_residual with essential rows carrying g_dot(t)
    - jacobian(y) = spatial_jacobian with essential rows zeroed
    - mass_matrix() = mass with identity rows at essential DOFs

    as the injected row treatment defines them.

    and, for implicit steppers, the same ODE as a residual-form system
    of the discretization's stack operators (:meth:`implicit_system`),
    with the essential rows applied here: the discretization applies
    none. The constraint VALUES g(t) for the end-of-step system are
    applied externally by the BC-enforcing time residual wrapper.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        The system on a stack; states are the stack's vectors.
        Consistent or lumped mass is the discretization's choice.
    solver : LinearSolverProtocol
        Solves the implicit system's shifted Jacobians; pairs with the
        stack's format ``F``.
    row_treatment : RowTreatmentFactory, optional
        Builds the essential-row strategy from the constraint set,
        backend, lumping and raw mass. Defaults to
        :class:`~pyapprox.pde.galerkin.time_integration.neutralized_rows.NeutralizedRows`;
        :class:`~pyapprox.pde.galerkin.time_integration.neutralized_rows.SymmetricNeutralizedRows`
        gives the symmetric mass.

    Raises
    ------
    TypeError
        If the row treatment refuses the system (the symmetric one, for a
        consistent mass with time-varying data and no analytic g_dot).

    Examples
    --------
    >>> ode_residual = skfem_ode_residual(compose_galerkin_system(physics, bcs))
    >>> time_stepper = BackwardEulerHVP(ode_residual)
    """

    def __init__(
        self,
        discretization: GalerkinStackDiscretization[Array, Array, F],
        solver: LinearSolverProtocol[Array, F],
        row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
    ) -> None:
        if not isinstance(discretization, GalerkinStackDiscretization):
            raise TypeError(
                "discretization must be a GalerkinStackDiscretization, got "
                f"{type(discretization).__name__}"
            )
        if not isinstance(solver, LinearSolverProtocol):
            raise TypeError(
                "solver must satisfy LinearSolverProtocol, got "
                f"{type(solver).__name__}"
            )
        system = discretization.system()
        if not isinstance(system, GalerkinTransientSystemProtocol):
            raise TypeError(
                "system must satisfy GalerkinTransientSystemProtocol, "
                f"got {type(system).__name__}"
            )
        self._system = system
        self._discretization = discretization
        self._solver = solver
        self._spatial_operator = system.spatial_operator()
        self._bkd = system.bkd()
        # Read by the parameterized subclasses' derivative constraints.
        self._constraint_set = system.constraint_set()
        # The lumping is the discretization's mass assembler's choice; the
        # matrix-era mass follows it, so the two masses agree.
        self._lumped_mass = discretization.assemble_mass().properties().diagonal
        self._rows: RowTreatmentProtocol[Array] = row_treatment(
            self._constraint_set, self._bkd, self._lumped_mass, system.mass_matrix()
        )
        self._mass = self._rows.mass()
        self._essential_jacobian = GalerkinJacobianOperator(
            discretization, discretization.essential_rows()
        )
        # The state an affine physics' Jacobian is assembled at: it ignores
        # the state, so any one gives the same Jacobian; allocated once.
        self._reference_state = self._bkd.zeros((system.nstates(),))
        self._stack_mass_source: Optional[ActionOperatorProtocol[Array, F]] = None
        self._stack_mass: Optional[ActionOperatorProtocol[Array, F]] = None
        self._derivatives: Optional[ODEResidualDerivatives[Array, F]] = None

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def derivatives(self) -> ODEResidualDerivatives[Array, F]:
        """The derivatives of the BC-neutralized ``f`` as operators, built
        on first request (a parameterized tier needs its parameterization
        bound first) and kept: the fields read the adapter's current
        parameters, so the bundle stays valid after ``set_param``.

        The Jacobian is the stack operator with the essential rows zeroed,
        in the discretization's format, the same operator the implicit
        system solves with; it equals :meth:`jacobian` under either row
        treatment, whose essential-row terms do not depend on the state."""
        if self._derivatives is None:
            self._derivatives = self._build_derivatives()
        return self._derivatives

    def _build_derivatives(self) -> ODEResidualDerivatives[Array, F]:
        """This tier's bundle, the state Jacobian only; each tier with
        parameter derivatives overrides it."""
        return ODEResidualDerivatives.state_only(self._jacobian())

    def _jacobian(self) -> Jacobian[Array, F]:
        """The stack Jacobian, essential rows zeroed, at what it depends
        on: of the time only when the physics declares :math:`F` affine in
        the state (its zero-curvature ``StateDerivatives``), so it is not
        given the state, and of neither when every part of :math:`F` also
        declares its Jacobian time-invariant. The essential rows carry the
        boundary velocity, independent of the state, so they keep the
        neutralized :math:`f` affine; the rows zeroed are fixed, so they
        keep its Jacobian time-invariant."""
        if not self._spatial_operator.state_derivatives().affine():
            return Jacobian.general(self._essential_jacobian.jacobian_operator)
        if self._spatial_operator.jacobian_is_time_invariant():
            return Jacobian.constant(self._constant_jacobian)
        return Jacobian.state_free(self._state_free_jacobian)

    def _state_free_jacobian(
        self, time: float, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[Array, F]:
        """The Jacobian of an affine :math:`F` at ``time``, assembled at the
        reference state: an affine physics ignores the state, so any state
        gives this same Jacobian, and zero is a convenient one."""
        return self._essential_jacobian.jacobian_operator(
            self._reference_state, time, point
        )

    def _constant_jacobian(
        self, point: LinearizationPoint
    ) -> JacobianSnapshotProtocol[Array, F]:
        """The Jacobian of an affine :math:`F` whose Jacobian is declared
        time-invariant: the same at every state and time, so assembled at
        the reference state and time zero."""
        return self._state_free_jacobian(0.0, point)

    def system(self) -> GalerkinTransientSystemProtocol[Array]:
        """Return the adapted system."""
        return self._system

    def lumped_mass(self) -> bool:
        """Whether the mass is lumped (diagonal), as the discretization's
        mass assembler declares."""
        return self._lumped_mass

    def __call__(self, state: Array, time: float) -> Array:
        """Evaluate the BC-neutralized residual f(y, t).

        Essential rows carry the boundary velocity g_dot(t) (exact
        zeros for time-invariant sets), and with the symmetric row
        treatment interior rows carry -M_ID g_dot(t), so stage solves
        against the BC-neutralized mass are the constrained stage systems.
        Without an analytic g_dot (permitted by the default treatment) the
        essential rows are zeroed.

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

    def _neutralized_stack_mass(self) -> ActionOperatorProtocol[Array, F]:
        """The stack mass with the row treatment's essential rows, rebuilt
        only when the discretization's mass changes."""
        assembled = self._discretization.assemble_mass()
        if self._stack_mass is None or self._stack_mass_source is not assembled:
            self._stack_mass = self._rows.stack_mass(
                assembled, self._discretization.essential_rows()
            )
            self._stack_mass_source = assembled
        return self._stack_mass

    def implicit_system(self) -> MassMatrixSystem[Array, F]:
        """The residual-form system implicit steppers solve with: the
        BC-neutralized ODE as stack operators (the row treatment's mass,
        zeroed essential rows in the Jacobian) and the adapter's solver."""
        return MassMatrixSystem(
            self,
            self._neutralized_stack_mass(),
            self._essential_jacobian,
            self._discretization.space(),
            self._solver,
        )

    def pinned_rows(self) -> EssentialPinnedRows[Array]:
        """The essential rows and their data, for pinning inside a
        stepper's solves (:class:`~pyapprox.ode.pinned_rows.RowPinning`;
        at the step end, the strategy the BC-enforcing wrapper implements
        on the old steppers) or after each step
        (:class:`~pyapprox.ode.pinned_rows.PinnedRowsConstraint`)."""
        return EssentialPinnedRows(self._discretization)

    def mass_form_system(self) -> MassFormSystem[Array, Array, F]:
        """The same BC-neutralized ODE as the steppers of
        :mod:`pyapprox.ode.runge_kutta` and
        :mod:`pyapprox.ode.linear_multistep` read it: this adapter as
        :math:`f` and its derivatives, the row treatment's stack mass, the
        bundle's Jacobian, the discretization's space, the adapter's solver,
        and the reshape between the stack's columns and 1-D states.

        The mass is the one assembled now; a discretization whose mass
        changes afterwards needs a new system.

        Raises
        ------
        TypeError
            If the discretization's space has no inner product.
        """
        space = self._discretization.space()
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "the discretization's space must be an inner-product space, "
                f"got {type(space).__name__}"
            )
        return MassFormSystem(
            rhs=self,
            mass=self._neutralized_stack_mass(),
            space=space,
            solver=self._solver,
            form=BackendColumnForm(self._bkd),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(system={self._system!r}, "
            f"solver={self._solver!r})"
        )
