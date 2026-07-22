"""BC-enforcing time residual adapter for Galerkin methods.

Wraps time stepping residuals and applies the physics' essential
constraints (via its cached ``DirichletConstraintSet``) to the residual,
Jacobian, and sensitivity quantities after the stepper assembles the
raw Newton system. Mirrors the collocation wrapper family
(``pde/collocation/time_integration/bc_time_residual_adapter.py``) but
is implemented on the constraint set with sparse-aware operations.

Class Hierarchy
---------------
GalerkinBCEnforcingForwardResidual
    Wraps SensitivityStepperProtocol: forward solve + sensitivity.

Adjoint and HVP tiers follow in later phases of the time-integration
refactor. Use ``create_galerkin_bc_enforcing_residual()`` to create the
appropriate wrapper for an inner stepper.
"""

from typing import Generic, Optional

from scipy.sparse import issparse

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    MatrixOperator,
    SparseMatrixOperator,
)
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import (
    SensitivityStepperProtocol,
    TimeSteppingResidualProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse


class GalerkinBCEnforcingForwardResidual(Generic[Array]):
    """Wraps a SensitivityStepperProtocol and applies essential BCs.

    Provides forward solve methods (bkd, bind, __call__, jacobian,
    linsolve) and sensitivity methods (is_explicit,
    has_prev_state_hessian, sensitivity_off_diag_jacobian,
    native_residual). Constrained rows are replaced with
    ``R[d] = y[d] - g(t_{n+1})`` and identity Jacobian rows through the
    physics' cached constraint set.

    Parameters
    ----------
    time_residual : TimeSteppingResidualProtocol
        The underlying time stepping residual. Must satisfy
        ``SensitivityStepperProtocol`` at runtime (the static type
        matches ``create_stepper``'s base-tier return, mirroring the
        stepper table's lazy capability-narrowing convention).
    physics : GalerkinPhysicsProtocol
        Galerkin physics providing ``constraint_set()``.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: TimeSteppingResidualProtocol[Array],
        physics: GalerkinPhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(time_residual, SensitivityStepperProtocol):
            raise TypeError(
                "time_residual must satisfy SensitivityStepperProtocol, "
                f"got {type(time_residual).__name__}"
            )
        if not isinstance(physics, GalerkinPhysicsProtocol):
            raise TypeError(
                "physics must satisfy GalerkinPhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._inner: SensitivityStepperProtocol[Array] = time_residual
        self._physics = physics
        self._bkd = bkd
        self._constraint_set = physics.constraint_set()
        self._t_np1 = 0.0
        # Constant-Jacobian operator cache for one-step-solvable
        # (explicit) steppers: J = BC-modified M, constant across steps,
        # so the factorization is built once and reused.
        self._constant_jacobian_op: Optional[
            LinearOperatorProtocol[Array]
        ] = None

    # -- TimeSteppingResidualProtocol --

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def bind(self, ctx: StepContext[Array]) -> None:
        """Bind the step context and track t_{n+1}."""
        self._inner.bind(ctx)
        self._t_np1 = ctx.t_curr

    def __call__(self, state: Array) -> Array:
        """Evaluate residual with constrained rows replaced at t_{n+1}."""
        residual = self._inner(state)
        return self._constraint_set.apply_to_residual(
            residual, state, self._t_np1
        )

    def jacobian(self, state: Array) -> Array:
        """Compute Jacobian with constrained rows replaced by identity."""
        jacobian = self._inner.jacobian(state)
        return self._constraint_set.apply_to_jacobian(jacobian)

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve J dy = residual using the BC-modified Jacobian.

        For one-step-solvable (explicit) steppers the BC-modified
        Jacobian is the constant BC-modified mass matrix, so its
        factorization is built once on first call and reused for every
        step. Implicit steppers re-assemble a state-dependent Jacobian
        each call.
        """
        if self._inner.is_one_step_solvable():
            if self._constant_jacobian_op is None:
                jacobian = self.jacobian(state)
                if issparse(jacobian) and isinstance(self._bkd, NumpyBkd):
                    self._constant_jacobian_op = SparseMatrixOperator(
                        jacobian, self.bkd()
                    )
                else:
                    self._constant_jacobian_op = MatrixOperator(
                        jacobian, self._bkd
                    )
            return self._constant_jacobian_op.solve(residual)
        return solve_maybe_sparse(
            self._bkd, self.jacobian(state), residual
        )

    def is_one_step_solvable(self) -> bool:
        """Constraint rows are linear in y, so this delegates unchanged."""
        return bool(self._inner.is_one_step_solvable())

    def is_multistage(self) -> bool:
        """Whether the wrapped scheme forms internal stage states."""
        return self._inner.is_multistage()

    # -- SensitivityStepperProtocol --

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Access the underlying ODE residual."""
        return self._inner.native_residual

    def is_explicit(self) -> bool:
        """Return whether the scheme is explicit."""
        return bool(self._inner.is_explicit())

    def has_prev_state_hessian(self) -> bool:
        """Return whether R_{n+1} depends on f(y_n)."""
        return bool(self._inner.has_prev_state_hessian())

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """Forward sensitivity off-diagonal block with BC rows zeroed.

        B_n[d,:] = 0 at constrained DOFs (constraint rows depend only on
        the current state). The inner stepper returns the raw -M.
        """
        result = self._inner.sensitivity_off_diag_jacobian(ctx, y_curr)
        return self._constraint_set.zero_rows(result)

    def zero_adjoint_rhs(
        self, dqdu: Array, zero_essential: bool = True
    ) -> Array:
        """Zero dQ/dy at essential (Dirichlet) DOFs.

        Parameters
        ----------
        dqdu : Array
            Functional derivative dQ/dy at a single time step.
            Shape: (nstates,).
        zero_essential : bool, default True
            If True, zero dQ/dy at essential DOFs, forcing
            lambda[d] = 0 — correct when differentiating w.r.t. PDE
            parameters. Set to False for gradients w.r.t. BC
            parameters.

        Returns
        -------
        Array
            dQ/dy with essential DOFs zeroed. Shape: (nstates,).
        """
        if not zero_essential:
            return dqdu
        return self._constraint_set.zero_entries(dqdu)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={type(self._inner).__name__}, "
            f"physics={type(self._physics).__name__}, "
            f"constraint_set={self._constraint_set!r})"
        )


def create_galerkin_bc_enforcing_residual(
    inner: TimeSteppingResidualProtocol[Array],
    physics: GalerkinPhysicsProtocol[Array],
    bkd: Backend[Array],
) -> GalerkinBCEnforcingForwardResidual[Array]:
    """Create a BC-enforcing wrapper for an inner stepper.

    Currently returns the forward-level wrapper; adjoint and HVP tiers
    are added in later phases of the time-integration refactor, at
    which point this factory narrows by protocol (most specific first)
    like collocation's ``create_bc_enforcing_residual``.

    Parameters
    ----------
    inner : TimeSteppingResidualProtocol
        The time stepping residual to wrap. Must satisfy
        ``SensitivityStepperProtocol`` at runtime.
    physics : GalerkinPhysicsProtocol
        Galerkin physics providing ``constraint_set()``.
    bkd : Backend
        Computational backend.

    Returns
    -------
    GalerkinBCEnforcingForwardResidual
        The BC-enforcing wrapper.

    Raises
    ------
    TypeError
        If the stepper is stage-based, the mass matrix is consistent
        (not diagonal/identity), and any time-varying essential BC
        lacks an analytic ``constrained_values_time_derivative``.
        Checked eagerly here, not at the first stage solve.
    """
    if inner.is_multistage() and isinstance(
        inner, SensitivityStepperProtocol
    ):
        mass = inner.native_residual.mass_matrix()
        if not mass.is_diagonal():
            missing = physics.constraint_set().missing_time_derivative_bcs()
            if missing:
                raise TypeError(
                    "stage-based stepper "
                    f"{type(inner).__name__} with a consistent mass "
                    "matrix requires an analytic boundary velocity "
                    "(constrained_values_time_derivative) on every "
                    "time-varying essential BC, but these lack it: "
                    f"{missing}. A consistent mass couples boundary "
                    "motion into interior rows via M_id*g_dot; "
                    "dropping or FD-approximating it silently corrupts "
                    "stage slopes. Remedies: (a) supply the analytic "
                    "derivative on each BC, (b) use lumped mass "
                    "(config.lumped_mass=True), or (c) use a one-step "
                    "implicit method (backward_euler/crank_nicolson), "
                    "whose difference quotient supplies the term "
                    "exactly."
                )
    return GalerkinBCEnforcingForwardResidual(inner, physics, bkd)
