"""
Protocols for time stepping residuals.

These protocols define the interface for time discretization schemes
(Forward Euler, Backward Euler, Crank-Nicolson, etc.) at different
capability levels.

Protocol Hierarchy
------------------
TimeSteppingResidualProtocol
    Basic time step R(y_n) = 0.
SensitivityStepperProtocol
    Adds sensitivity methods (is_explicit, sensitivity_off_diag_jacobian).
AdjointEnabledTimeSteppingResidualProtocol
    Adds adjoint methods for gradient computation.
HVPEnabledTimeSteppingResidualProtocol
    Adds 4 same-step + 3 cross-step HVP methods.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Protocol for discretized time step residuals.

    This is the framework-level protocol for time stepping schemes like
    Forward Euler, Backward Euler, Crank-Nicolson, etc.

    The residual R(y_n) = 0 represents one time step from y_{n-1} to y_n.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def bind(self, ctx: StepContext[Array]) -> None:
        """
        Bind the step context for Newton-facing methods.

        This must be called before __call__ or jacobian.

        Parameters
        ----------
        ctx : StepContext
            Immutable context describing the current step.
        """
        ...

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the time stepping residual R(y_n).

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual R(y_n). Shape: (nstates,)
        """
        ...

    def jacobian(self, state: Array) -> Array:
        """
        Compute the Jacobian dR/dy_n.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian dR/dy_n. Shape: (nstates, nstates)
        """
        ...

    def linsolve(self, state: Array, residual: Array) -> Array:
        """
        Solve the linear system J dy = residual.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)
        residual : Array
            Right-hand side. Shape: (nstates,)

        Returns
        -------
        Array
            Solution dy. Shape: (nstates,)
        """
        ...

    def is_one_step_solvable(self) -> bool:
        """Return True if R(y_n) = 0 is linear in y_n with constant Jacobian.

        When True, one Newton step is exact (no iteration needed).
        True for explicit steppers (Forward Euler, Heun) where the
        Jacobian is the mass matrix alone. False for implicit steppers
        (Backward Euler, Crank-Nicolson) where f(y_n) introduces
        state-dependent nonlinearity.
        """
        ...


@runtime_checkable
class SensitivityStepperProtocol(Protocol, Generic[Array]):
    """
    Time stepping residual with sensitivity support.

    Extends TimeSteppingResidualProtocol with methods needed for forward
    sensitivity propagation and basic adjoint infrastructure.
    """

    def bkd(self) -> Backend[Array]: ...

    def bind(self, ctx: StepContext[Array]) -> None: ...

    def __call__(self, state: Array) -> Array: ...

    def jacobian(self, state: Array) -> Array: ...

    def linsolve(self, state: Array, residual: Array) -> Array: ...

    def is_one_step_solvable(self) -> bool: ...

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Return the underlying ODE residual."""
        ...

    def is_explicit(self) -> bool:
        """Return True if the time stepping scheme is explicit."""
        ...

    def has_prev_state_hessian(self) -> bool:
        """Return True if R_{n+1} depends on f(y_n)."""
        ...

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """Compute dR_n/dy_{n-1} for forward sensitivity propagation.

        Parameters
        ----------
        ctx : StepContext
            Context for this step (t_prev, deltat, y_prev).
        y_curr : Array
            Solution at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Off-diagonal Jacobian dR_n/dy_{n-1}. Shape: (nstates, nstates)
        """
        ...


@runtime_checkable
class AdjointEnabledTimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Time stepping residual with adjoint capability for gradient computation.

    Extends SensitivityStepperProtocol with methods required for the
    adjoint method to compute dQ/dp.

    Post-hoc methods take their context as explicit parameters — they
    never read bound state from the stepper.
    """

    def bkd(self) -> Backend[Array]: ...

    def bind(self, ctx: StepContext[Array]) -> None: ...

    def __call__(self, state: Array) -> Array: ...

    def jacobian(self, state: Array) -> Array: ...

    def linsolve(self, state: Array, residual: Array) -> Array: ...

    def is_one_step_solvable(self) -> bool: ...

    @property
    def native_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Return the underlying ODE residual."""
        ...

    def is_explicit(self) -> bool: ...

    def has_prev_state_hessian(self) -> bool: ...

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array: ...

    def param_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        Parameters
        ----------
        ctx : StepContext
            Context for this step (t_prev, deltat, y_prev).
        y_curr : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian dR/dp. Shape: (nstates, nparams)
        """
        ...

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        """
        Return a LinearOperator representing (dR/dy_n)^T.

        Parameters
        ----------
        ctx : StepContext
            Context for this step.
        y_curr : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        LinearOperatorProtocol[Array]
            Operator whose .solve(rhs) solves (dR/dy_n)^T x = rhs.
        """
        ...

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        Parameters
        ----------
        next_ctx : StepContext
            Context for the NEXT step (t_prev=t_n, deltat=dt_{n+1},
            y_prev=y_n). next_ctx.y_prev is the state at which we
            differentiate.
        y_curr_of_next : Array
            Solution at time n+1 (needed for implicit midpoint).
            Shape: (nstates,)

        Returns
        -------
        Array
            Off-diagonal coupling term. Shape: (nstates, nstates)
        """
        ...

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute the initial condition for backward adjoint solve.

        Parameters
        ----------
        ctx : StepContext
            Context for the final step.
        final_fwd_sol : Array
            Forward solution at final time. Shape: (nstates,)
        final_dqdu : Array
            Gradient dQ/dy at final time. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution at final time lambda_N. Shape: (nstates,)
        """
        ...

    def adjoint_final_solution(
        self,
        ctx: StepContext[Array],
        next_ctx: StepContext[Array],
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
    ) -> Array:
        """
        Compute the adjoint at initial time (final step of backward sweep).

        Parameters
        ----------
        ctx : StepContext
            Context at t=0.
        next_ctx : StepContext
            Context for the first step (needed for off-diag).
        fsol_0 : Array
            Forward solution at initial time. Shape: (nstates,)
        asol_1 : Array
            Adjoint solution at time step 1. Shape: (nstates,)
        dqdu_0 : Array
            Gradient dQ/dy at initial time. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution at initial time lambda_0. Shape: (nstates,)
        """
        ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature rule consistent with time discretization.

        Parameters
        ----------
        times : Array
            Time points. Shape: (ntimes,)

        Returns
        -------
        quadx : Array
            Quadrature sample points. Shape: (nquad,)
        quadw : Array
            Quadrature weights. Shape: (nquad,)
        """
        ...

    def zero_adjoint_rhs(self, dqdu: Array) -> Array:
        """Zero dQ/dy entries at constrained DOFs for adjoint solve."""
        ...

    def initial_param_jacobian(self) -> Array:
        """Jacobian of initial condition w.r.t. parameters.

        Returns d(y_0)/dp. Shape: (nstates, nparams).
        """
        ...


@runtime_checkable
class HVPEnabledTimeSteppingResidualProtocol(Protocol, Generic[Array]):
    """
    Time stepping residual with HVP capability for second-order adjoints.

    Extends AdjointEnabledTimeSteppingResidualProtocol with four same-step
    and three cross-step HVP methods. All steppers implement prev_*
    (BE returns zero, FE/Heun delegate).
    """

    # All AdjointEnabledTimeSteppingResidualProtocol methods
    def bkd(self) -> Backend[Array]: ...

    def bind(self, ctx: StepContext[Array]) -> None: ...

    def __call__(self, state: Array) -> Array: ...

    def jacobian(self, state: Array) -> Array: ...

    def linsolve(self, state: Array, residual: Array) -> Array: ...

    def is_one_step_solvable(self) -> bool: ...

    @property
    def native_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]: ...

    def is_explicit(self) -> bool: ...

    def has_prev_state_hessian(self) -> bool: ...

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array: ...

    def param_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array: ...

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]: ...

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array: ...

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array: ...

    def adjoint_final_solution(
        self,
        ctx: StepContext[Array],
        next_ctx: StepContext[Array],
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
    ) -> Array: ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]: ...

    def zero_adjoint_rhs(self, dqdu: Array) -> Array: ...

    def initial_param_jacobian(self) -> Array: ...

    # Same-step HVP methods

    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array: ...

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array: ...

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array: ...

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array: ...

    # Cross-step HVP methods

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array: ...

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array: ...

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array: ...
