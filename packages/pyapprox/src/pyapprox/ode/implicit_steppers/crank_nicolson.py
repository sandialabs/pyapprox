r"""Crank-Nicolson time stepping residual with adjoint support.

The Crank-Nicolson method is a second-order implicit time integrator:

.. math::

    M (y_n - y_{n-1}) - \frac{\Delta t}{2}
    \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr] = 0

Split into three classes via mixin composition:

- CrankNicolsonStepper: core + sensitivity + quadrature + implicit
- CrankNicolsonAdjoint: + adjoint methods
- CrankNicolsonHVP: + HVP methods (same-step + cross-step)
"""

from typing import Generic

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    TransposeLinearOperator,
)
from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin
from pyapprox.ode.mixins.implicit import ImplicitStepperMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin
from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
    ImplicitODEResidualWithHVPProtocol,
    ImplicitODEResidualWithParamJacobianProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array

# =========================================================================
# Base stepper: core + sensitivity + quadrature + implicit
# =========================================================================


class CrankNicolsonStepper(
    SensitivityMixin[Array],
    ImplicitStepperMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Crank-Nicolson time stepping residual (base level).

    Second-order implicit method (A-stable):

    .. math::

        R(y_n) = M (y_n - y_{n-1})
        - \frac{\Delta t}{2}
        \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr] = 0
    """

    _residual: ImplicitODEResidualProtocol[Array]

    def __init__(self, residual: ImplicitODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def _newton_coefficient(self) -> float:
        return 0.5 * self._ctx.deltat

    def __call__(self, state: Array) -> Array:
        # f(y_{n-1}, t_{n-1})
        self._residual.set_time(self._ctx.t_prev)
        current_res = self._residual(self._ctx.y_prev)

        # f(y_n, t_n)
        self._residual.set_time(self._ctx.t_curr)
        next_res = self._residual(state)

        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - 0.5 * self._ctx.deltat * (current_res + next_res)

    def jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dR/dy_n = M - (\Delta t/2) \, (df/dy)|_{y_n}`."""
        self._residual.set_time(self._ctx.t_curr)
        return self._residual.newton_jacobian(
            state, self._newton_coefficient()
        ).as_matrix()

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return False

    def is_one_step_solvable(self) -> bool:
        return False

    def has_prev_state_hessian(self) -> bool:
        return True

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1} = -(M + (\Delta t/2) \, J_{n-1})`.

        .. math::

            \frac{dR_n}{dy_{n-1}}
            = -\bigl(M + \tfrac{\Delta t}{2} \, J_{n-1}\bigr)

        where :math:`J_{n-1} = (df/dy)|_{y_{n-1}}`.
        """
        self._residual.set_time(ctx.t_prev)
        mass = self._residual.mass_matrix().as_matrix()
        jac = self._residual.jacobian(ctx.y_prev)
        return -(mass + 0.5 * ctx.deltat * jac)

    # -- QuadratureMixin --

    def quadrature_samples_weights(self, times: Array) -> tuple:
        """Trapezoidal quadrature (nodes, trapezoidal weights)."""
        weights = self._bkd.zeros(times.shape)
        for ii in range(times.shape[0]):
            if ii > 0:
                weights[ii] = weights[ii] + 0.5 * (times[ii] - times[ii - 1])
            if ii < times.shape[0] - 1:
                weights[ii] = weights[ii] + 0.5 * (times[ii + 1] - times[ii])
        return times, weights


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class CrankNicolsonAdjoint(
    AdjointMixin[Array],
    CrankNicolsonStepper[Array],
    Generic[Array],
):
    """Crank-Nicolson with adjoint capability for gradient computation."""

    _residual: ImplicitODEResidualWithParamJacobianProtocol[Array]

    def __init__(
        self, residual: ImplicitODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR/dp`.

        .. math::

            \frac{dR}{dp} = -\frac{\Delta t}{2}
            \Bigl[\frac{df}{dp}\Big|_{y_{n-1}, t_{n-1}}
            + \frac{df}{dp}\Big|_{y_n, t_n}\Bigr]
        """
        self._residual.set_time(ctx.t_prev)
        current_param_jac = self._adjoint_residual.param_jacobian(ctx.y_prev)

        self._residual.set_time(ctx.t_curr)
        next_param_jac = self._adjoint_residual.param_jacobian(y_curr)

        return -0.5 * ctx.deltat * (current_param_jac + next_param_jac)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        r"""Compute :math:`(dR/dy_n)^T = (M - (\Delta t/2) \, J)^T`."""
        self._residual.set_time(ctx.t_curr)
        op = self._residual.newton_jacobian(
            y_curr, 0.5 * ctx.deltat
        )
        return TransposeLinearOperator(op)

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T`.

        .. math::

            \Bigl(\frac{dR_{n+1}}{dy_n}\Bigr)^T
            = -\bigl(M + \tfrac{\Delta t_{n+1}}{2} \, J_n\bigr)^T
        """
        self._residual.set_time(next_ctx.t_prev)
        return -(
            self._residual.mass_matrix().as_matrix()
            + 0.5 * next_ctx.deltat * self._residual.jacobian(next_ctx.y_prev)
        ).T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        self._residual.set_time(ctx.t_curr)
        op = self._residual.newton_jacobian(
            final_fwd_sol, 0.5 * ctx.deltat
        )
        return op.solve_transpose(-final_dqdu)


# =========================================================================
# HVP level: + HVP methods (same-step + cross-step)
# =========================================================================


class CrankNicolsonHVP(
    HVPMixin[Array],
    CrankNicolsonAdjoint[Array],
    Generic[Array],
):
    r"""Crank-Nicolson with HVP capability for Hessian-vector products.

    The residual is:

    .. math::

        R(y_n) = M (y_n - y_{n-1})
        - \frac{\Delta t}{2}
        \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr]

    Key insight: :math:`y_{n-1}` is fixed when differentiating w.r.t.
    :math:`y_n`, so the :math:`f(y_{n-1})` term drops from
    :math:`d^2 R / dy_n^2`.  However, both terms contribute to
    :math:`d^2 R / dp^2` because both depend on :math:`p`.

    The ``prev_*`` methods compute the :math:`R_{n+1}` contribution
    evaluated at :math:`y_n` (which acts as :math:`y_{n-1}` for
    :math:`R_{n+1}`). They use ``next_ctx.deltat`` (= dt_{n+1}), fixing
    the non-uniform dt bug in the old implementation.
    """

    _residual: ImplicitODEResidualWithHVPProtocol[Array]

    def __init__(self, residual: ImplicitODEResidualWithHVPProtocol[Array]) -> None:
        super().__init__(residual)

    # -- Same-step HVP methods --

    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n^2) w = -(\Delta t/2)
        \, (d^2f/dy^2)|_{y_n} \, w`."""
        self._residual.set_time(ctx.t_curr)
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.state_state_hvp(y_curr, adj_state, wvec)
        )

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n \, dp) v`."""
        self._residual.set_time(ctx.t_curr)
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.state_param_hvp(y_curr, adj_state, vvec)
        )

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy_n) w`."""
        self._residual.set_time(ctx.t_curr)
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.param_state_hvp(y_curr, adj_state, wvec)
        )

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp^2) v`."""
        # Contribution from y_{n-1} term
        self._residual.set_time(ctx.t_prev)
        hvp_nm1 = self._hvp_residual.param_param_hvp(
            ctx.y_prev, adj_state, vvec
        )

        # Contribution from y_n term
        self._residual.set_time(ctx.t_curr)
        hvp_n = self._hvp_residual.param_param_hvp(y_curr, adj_state, vvec)

        return -0.5 * ctx.deltat * (hvp_nm1 + hvp_n)

    # -- Cross-step HVP methods --
    # Use next_ctx.deltat (= dt_{n+1}), not self._ctx.deltat (= dt_n).
    # This is the structural fix for the non-uniform dt bug.

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dy_n^2) w`."""
        self._residual.set_time(next_ctx.t_prev)
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.state_state_hvp(
                next_ctx.y_prev, adj_state, wvec
            )
        )

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dy_n \, dp) v`."""
        self._residual.set_time(next_ctx.t_prev)
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.state_param_hvp(
                next_ctx.y_prev, adj_state, vvec
            )
        )

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dp \, dy_n) w`."""
        self._residual.set_time(next_ctx.t_prev)
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.param_state_hvp(
                next_ctx.y_prev, adj_state, wvec
            )
        )
