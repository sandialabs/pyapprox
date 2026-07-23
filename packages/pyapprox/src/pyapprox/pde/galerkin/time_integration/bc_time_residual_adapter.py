"""BC-enforcing time residual adapter for Galerkin methods.

Wraps time stepping residuals and applies the physics' essential
constraints (via its cached ``DirichletConstraintSet``) to the residual,
Jacobian, and sensitivity/adjoint quantities after the stepper
assembles the raw Newton system. Mirrors the collocation wrapper family
(``pde/collocation/time_integration/bc_time_residual_adapter.py``) but
is implemented on the constraint set with sparse-aware operations.

Class Hierarchy
---------------
GalerkinBCEnforcingForwardResidual
    Wraps SensitivityStepperProtocol: forward solve + sensitivity.
GalerkinBCEnforcingAdjointResidual
    Wraps AdjointEnabledTimeSteppingResidualProtocol: + adjoint methods.
GalerkinBCEnforcingHVPResidual
    Wraps HVPEnabledTimeSteppingResidualProtocol: + all HVP methods
    (4 same-step + cross-step).

Use ``create_galerkin_bc_enforcing_residual()`` to create the widest
wrapper the inner stepper supports.
"""

from typing import Generic, Optional, Tuple, Union, overload

from scipy.sparse import issparse, spmatrix

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    MatrixOperator,
    SparseMatrixOperator,
    TransposeLinearOperator,
)
from pyapprox.ode.protocols.ode_residual import ODEResidualProtocol
from pyapprox.ode.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
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


class GalerkinBCEnforcingAdjointResidual(
    GalerkinBCEnforcingForwardResidual[Array], Generic[Array]
):
    """Extends the forward wrapper with adjoint methods.

    Wraps an AdjointEnabledTimeSteppingResidualProtocol. All Dirichlet
    handling is owned here (parameterizations return RAW dR/dp per the
    refactor design): constrained rows of parameter Jacobians are
    zeroed, the adjoint diagonal is the transpose of the BC-enforced
    forward Jacobian (sparse-factored when possible), and the adjoint
    off-diagonal has constrained COLUMNS zeroed (the transpose of the
    forward off-diagonal's zeroed rows).

    DAE masses (singular, e.g. Stokes) are not yet supported by the
    mass-only adjoint solves; that lands with the D6 adjoint work.
    """

    def __init__(
        self,
        time_residual: TimeSteppingResidualProtocol[Array],
        physics: GalerkinPhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(time_residual, physics, bkd)
        if not isinstance(
            time_residual, AdjointEnabledTimeSteppingResidualProtocol
        ):
            raise TypeError(
                f"{type(self).__name__} requires an adjoint-tier inner "
                f"stepper, got {type(time_residual).__name__}"
            )
        self._adjoint_inner: AdjointEnabledTimeSteppingResidualProtocol[
            Array
        ] = time_residual

    def _transposed_operator(
        self, matrix: Union[spmatrix, Array]
    ) -> LinearOperatorProtocol[Array]:
        """Wrap a matrix as its transpose operator, sparse-factored."""
        if issparse(matrix) and isinstance(self._bkd, NumpyBkd):
            return TransposeLinearOperator(
                SparseMatrixOperator(matrix, self.bkd())
            )
        return MatrixOperator(matrix.T, self._bkd)

    def param_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """Compute dR/dp with constrained rows zeroed.

        The inner stepper returns the raw parameter Jacobian; essential
        constraints are parameter-independent, so their rows vanish.
        """
        result = self._adjoint_inner.param_jacobian(ctx, y_curr)
        return self._constraint_set.zero_rows(result)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        """Adjoint diagonal block: transpose of the BC-enforced forward
        Jacobian, as a (sparse-factored where possible) operator.

        Binds the given step context first: the forward ``jacobian``
        reads bound state (deltat, t_{n+1}), which otherwise holds the
        LAST forward step's values during the backward sweep.
        """
        self.bind(ctx)
        return self._transposed_operator(self.jacobian(y_curr))

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        """Adjoint off-diagonal block: B_{n+1}^T with constrained
        COLUMNS zeroed (rows of B_{n+1} were replaced, so its transpose
        has zero columns there)."""
        result = self._adjoint_inner.adjoint_off_diag_jacobian(
            next_ctx, y_curr_of_next
        )
        return self._constraint_set.zero_cols(result)

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """Adjoint terminal condition via the BC-enforced Jacobian."""
        final_dqdu = self.zero_adjoint_rhs(final_dqdu)
        drdu_diag_t = self.adjoint_diag_jacobian(ctx, final_fwd_sol)
        return drdu_diag_t.solve(-final_dqdu)

    def adjoint_final_solution(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        asol_1: Array,
        dqdu_0: Array,
    ) -> Array:
        """Adjoint at the initial time via the BC-neutralized mass.

        The adapter's mass already carries identity rows at essential
        DOFs, so M^T has identity columns there and the solve pins
        lambda_0[d] correctly with dQ/dy zeroed at essential DOFs.
        """
        mass = self._adjoint_inner.native_residual.mass_matrix()
        if mass.is_singular():
            raise NotImplementedError(
                "adjoint_final_solution with a singular (DAE) mass "
                "matrix is not yet supported; it lands with the DAE "
                "adjoint work"
            )
        dqdu_0 = self.zero_adjoint_rhs(dqdu_0)
        drdu_offdiag_t = self.adjoint_off_diag_jacobian(ctx, y_curr)
        rhs = -self._matvec(drdu_offdiag_t, asol_1) - dqdu_0
        return mass.solve_transpose(rhs)

    def _matvec(
        self, matrix: Union[spmatrix, Array], vec: Array
    ) -> Array:
        """Sparse-aware matrix-vector product returning a backend array."""
        if issparse(matrix):
            return self._bkd.asarray(matrix @ self._bkd.to_numpy(vec))
        return self._bkd.dot(matrix, vec)

    def quadrature_samples_weights(
        self, times: Array
    ) -> Tuple[Array, Array]:
        """Quadrature rule consistent with the time discretization."""
        return self._adjoint_inner.quadrature_samples_weights(times)

    def initial_param_jacobian(self) -> Array:
        """d(initial_state)/dp with constrained rows zeroed."""
        result = self._adjoint_inner.initial_param_jacobian()
        return self._constraint_set.zero_rows(result)


class GalerkinBCEnforcingHVPResidual(
    GalerkinBCEnforcingAdjointResidual[Array], Generic[Array]
):
    """Extends the adjoint wrapper with all HVP methods.

    Wraps an HVPEnabledTimeSteppingResidualProtocol. Constraint rows
    are LINEAR in (y, p): their true second derivatives vanish, but the
    inner stepper contracts the RAW second-derivative tensors, whose
    constrained rows are generally nonzero. So EVERY contraction —
    same-step, param-shaped, AND cross-step — receives the adjoint with
    constrained entries zeroed. That is the ONLY correction: with
    lambda_d = 0 the contraction equals the true wrapped tensor
    sum_{j interior} lambda_j d2R_j c, so every method returns exact
    wrapped values (entrywise FD-checkable), including the generally
    NONZERO state-shaped entries at constrained indices d (interior
    rows genuinely depend on boundary DOFs).

    Those d-entries feed only the constrained rows of the second-order
    adjoint recursion, whose solution component s_d is decoupled from
    the HVP: the BC-enforced Jacobian has row d = e_d^T, so J^T has
    column d = e_d and s_d appears in no interior equation; and every
    consumer of s annihilates the d-entry (dR/dp rows zeroed,
    cross-step B rows zeroed, initial_param_jacobian rows zeroed).
    Correctness therefore rests on that invariant, not on masking
    outputs.
    """

    def __init__(
        self,
        time_residual: TimeSteppingResidualProtocol[Array],
        physics: GalerkinPhysicsProtocol[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(time_residual, physics, bkd)
        if not isinstance(
            time_residual, HVPEnabledTimeSteppingResidualProtocol
        ):
            raise TypeError(
                f"{type(self).__name__} requires an HVP-tier inner "
                f"stepper, got {type(time_residual).__name__}"
            )
        self._hvp_inner: HVPEnabledTimeSteppingResidualProtocol[Array] = (
            time_residual
        )

    def _zeroed_adjoint(self, adj_state: Array) -> Array:
        """Adjoint with constrained entries zeroed for RAW contractions."""
        return self._constraint_set.zero_entries(adj_state)

    # -- Same-step HVP methods --

    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R/dy_n^2) w with the constrained adjoint zeroed."""
        return self._hvp_inner.state_state_hvp(
            ctx, y_curr, self._zeroed_adjoint(adj_state), wvec
        )

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/dy_n dp) v with the constrained adjoint zeroed."""
        return self._hvp_inner.state_param_hvp(
            ctx, y_curr, self._zeroed_adjoint(adj_state), vvec
        )

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp dy_n) w with the constrained adjoint zeroed."""
        return self._hvp_inner.param_state_hvp(
            ctx, y_curr, self._zeroed_adjoint(adj_state), wvec
        )

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R/dp^2) v with the constrained adjoint zeroed."""
        return self._hvp_inner.param_param_hvp(
            ctx, y_curr, self._zeroed_adjoint(adj_state), vvec
        )

    # -- Cross-step HVP methods --

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R_{n+1}/dy_n^2) w with the constrained adjoint zeroed."""
        return self._hvp_inner.prev_state_state_hvp(
            next_ctx, y_curr_of_next, self._zeroed_adjoint(adj_state), wvec
        )

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """adj^T (d^2R_{n+1}/dy_n dp) v with the constrained adjoint zeroed."""
        return self._hvp_inner.prev_state_param_hvp(
            next_ctx, y_curr_of_next, self._zeroed_adjoint(adj_state), vvec
        )

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """adj^T (d^2R_{n+1}/dp dy_n) w with the constrained adjoint zeroed."""
        return self._hvp_inner.prev_param_state_hvp(
            next_ctx, y_curr_of_next, self._zeroed_adjoint(adj_state), wvec
        )

    def state_prev_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec_prev: Array,
    ) -> Array:
        """adj^T (d^2R_n/dy_n dy_{n-1}) w_prev, constrained adjoint zeroed."""
        return self._hvp_inner.state_prev_state_hvp(
            ctx, y_curr, self._zeroed_adjoint(adj_state), wvec_prev
        )

    def prev_state_curr_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec_curr_of_next: Array,
    ) -> Array:
        """adj^T (d^2R_{n+1}/dy_n dy_{n+1}) w, constrained adjoint zeroed."""
        return self._hvp_inner.prev_state_curr_state_hvp(
            next_ctx,
            y_curr_of_next,
            self._zeroed_adjoint(adj_state),
            wvec_curr_of_next,
        )


@overload
def create_galerkin_bc_enforcing_residual(
    inner: HVPEnabledTimeSteppingResidualProtocol[Array],
    physics: GalerkinPhysicsProtocol[Array],
    bkd: Backend[Array],
) -> GalerkinBCEnforcingHVPResidual[Array]: ...


@overload
def create_galerkin_bc_enforcing_residual(
    inner: AdjointEnabledTimeSteppingResidualProtocol[Array],
    physics: GalerkinPhysicsProtocol[Array],
    bkd: Backend[Array],
) -> GalerkinBCEnforcingAdjointResidual[Array]: ...


@overload
def create_galerkin_bc_enforcing_residual(
    inner: TimeSteppingResidualProtocol[Array],
    physics: GalerkinPhysicsProtocol[Array],
    bkd: Backend[Array],
) -> GalerkinBCEnforcingForwardResidual[Array]: ...


def create_galerkin_bc_enforcing_residual(
    inner: TimeSteppingResidualProtocol[Array],
    physics: GalerkinPhysicsProtocol[Array],
    bkd: Backend[Array],
) -> GalerkinBCEnforcingForwardResidual[Array]:
    """Create the widest BC-enforcing wrapper the inner stepper supports.

    Narrows by protocol, most specific first (HVP -> Adjoint ->
    Forward), like collocation's ``create_bc_enforcing_residual``.

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
        The widest BC-enforcing wrapper (may be a subclass).

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
    if isinstance(inner, HVPEnabledTimeSteppingResidualProtocol):
        return GalerkinBCEnforcingHVPResidual(inner, physics, bkd)
    if isinstance(inner, AdjointEnabledTimeSteppingResidualProtocol):
        return GalerkinBCEnforcingAdjointResidual(inner, physics, bkd)
    return GalerkinBCEnforcingForwardResidual(inner, physics, bkd)
