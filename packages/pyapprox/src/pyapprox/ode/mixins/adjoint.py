"""Adjoint mixin for gradient computation via the discrete adjoint method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


class AdjointMixin(ABC, Generic[Array]):
    """Mixin providing adjoint methods for gradient computation.

    Requires the underlying ODE residual to support param_jacobian
    (ODEResidualWithParamJacobianProtocol). This is enforced by the
    concrete stepper's __init__ accepting the narrower type.

    Uses _adjoint_residual property for typed access to the narrowed
    residual, avoiding field redeclaration conflicts with CoreStepperMixin.
    """

    if TYPE_CHECKING:
        _residual: ODEResidualProtocol[Array]
        _bkd: Backend[Array]
        _ctx: StepContext[Array]

    @property
    def _adjoint_residual(
        self,
    ) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Typed access to the ODE residual as param-jacobian capable.

        Safe because concrete stepper __init__ enforces this type.
        """
        return cast(
            ODEResidualWithParamJacobianProtocol[Array], self._residual
        )

    @property
    def native_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Get the underlying ODE residual (narrowed to param jacobian capable)."""
        return self._adjoint_residual

    @abstractmethod
    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """Compute the parameter Jacobian dR/dp for one time step."""
        ...

    def param_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        """Compute the parameter Jacobian dR/dp for one time step."""
        return self._param_jacobian_impl(ctx, y_curr)

    @abstractmethod
    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        """Compute the diagonal Jacobian block for adjoint solve: (dR/dy_n)^T."""
        ...

    @abstractmethod
    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        """Compute the off-diagonal Jacobian for adjoint coupling."""
        ...

    @abstractmethod
    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """Compute the initial condition for backward adjoint solve."""
        ...

    def zero_adjoint_rhs(self, dqdu: Array) -> Array:
        """Zero dQ/dy entries at constrained DOFs for adjoint solve.

        Default no-op: returns dqdu unchanged. BC-enforcing wrappers
        override to zero essential (Dirichlet) DOFs.
        """
        return dqdu

    def initial_param_jacobian(self) -> Array:
        """Jacobian of initial condition w.r.t. parameters.

        Default delegates to native_residual.initial_param_jacobian().
        """
        return self._adjoint_residual.initial_param_jacobian()

    def adjoint_final_solution(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        asol_1: Array,
        dqdu_0: Array,
    ) -> Array:
        """Compute the adjoint at initial time (final step of backward sweep).

        Solves: M^T lambda_0 = -B_1^T lambda_1 - dQ/dy_0

        Symmetric with adjoint_initial_condition: ctx describes R_1
        (the first forward step), ctx.y_prev = y_0, y_curr = y_1.
        """
        mass = self._adjoint_residual.mass_matrix()
        drduT_offdiag = self.adjoint_off_diag_jacobian(ctx, y_curr)
        rhs = -drduT_offdiag @ asol_1 - dqdu_0
        return mass.solve_transpose(rhs)
