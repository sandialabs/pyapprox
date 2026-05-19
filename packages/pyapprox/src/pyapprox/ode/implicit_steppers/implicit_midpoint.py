r"""Implicit midpoint time stepping residual with adjoint support.

The implicit midpoint method is a second-order implicit time integrator:

.. math::

    M (y_n - y_{n-1}) - \Delta t \, f\!\bigl(
    y_{\mathrm{mid}},\, t_{\mathrm{mid}}\bigr) = 0

where :math:`y_{\mathrm{mid}} = (y_n + y_{n-1})/2` and
:math:`t_{\mathrm{mid}} = t_{n-1} + \Delta t/2`.

It is A-stable and symplectic (preserves quadratic invariants).

Split into three classes via mixin composition:

- ImplicitMidpointStepper: core + sensitivity + quadrature + implicit
- ImplicitMidpointAdjoint: + adjoint methods
- ImplicitMidpointHVP: + HVP methods (same-step + cross-step)
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


class ImplicitMidpointStepper(
    SensitivityMixin[Array],
    ImplicitStepperMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Implicit midpoint time stepping residual (base level).

    Second-order implicit method (A-stable, symplectic):

    .. math::

        R(y_n) = M (y_n - y_{n-1})
        - \Delta t \, f\!\bigl(\tfrac{y_n + y_{n-1}}{2},\,
        t_{n-1} + \tfrac{\Delta t}{2}\bigr) = 0
    """

    _residual: ImplicitODEResidualProtocol[Array]

    def __init__(self, residual: ImplicitODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def _newton_coefficient(self) -> float:
        return 0.5 * self._ctx.deltat

    def _set_midpoint_time(self, ctx: StepContext[Array]) -> None:
        self._residual.set_time(ctx.t_prev + 0.5 * ctx.deltat)

    def _midpoint_state(self, ctx: StepContext[Array], y_curr: Array) -> Array:
        return 0.5 * (ctx.y_prev + y_curr)

    def __call__(self, state: Array) -> Array:
        y_mid = self._midpoint_state(self._ctx, state)
        self._set_midpoint_time(self._ctx)
        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - self._ctx.deltat * self._residual(y_mid)

    def jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dR/dy_n = M - (\Delta t/2) \, J(y_{\mathrm{mid}})`.

        Chain rule: :math:`dy_{\mathrm{mid}}/dy_n = 1/2`.
        """
        y_mid = self._midpoint_state(self._ctx, state)
        self._set_midpoint_time(self._ctx)
        return self._residual.newton_jacobian(
            y_mid, self._newton_coefficient()
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
        r"""Compute :math:`dR_n/dy_{n-1} = -(M + (\Delta t/2) \, J(y_{\mathrm{mid}}))`.

        Chain rule: :math:`dy_{\mathrm{mid}}/dy_{n-1} = 1/2`.
        """
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        mass = self._residual.mass_matrix().as_matrix()
        jac = self._residual.jacobian(y_mid)
        return -(mass + 0.5 * ctx.deltat * jac)

    # -- QuadratureMixin --

    def quadrature_samples_weights(self, times: Array) -> tuple:
        """Midpoint quadrature (interval midpoints, interval widths)."""
        return (times[:-1] + times[1:]) / 2, self._bkd.diff(times)


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class ImplicitMidpointAdjoint(
    AdjointMixin[Array],
    ImplicitMidpointStepper[Array],
    Generic[Array],
):
    """Implicit midpoint with adjoint capability for gradient computation."""

    _residual: ImplicitODEResidualWithParamJacobianProtocol[Array]

    def __init__(
        self, residual: ImplicitODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR/dp = -\Delta t
        \, (df/dp)|_{y_{\mathrm{mid}}, t_{\mathrm{mid}}}`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return -ctx.deltat * self._adjoint_residual.param_jacobian(y_mid)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        r"""Compute :math:`(dR/dy_n)^T = (M - (\Delta t/2)
        \, J(y_{\mathrm{mid}}))^T`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        op = self._residual.newton_jacobian(y_mid, 0.5 * ctx.deltat)
        return TransposeLinearOperator(op)

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T`.

        .. math::

            \Bigl(\frac{dR_{n+1}}{dy_n}\Bigr)^T
            = -\bigl(M + \tfrac{\Delta t_{n+1}}{2} \, J(y_{\mathrm{mid},n+1})\bigr)^T

        where :math:`y_{\mathrm{mid},n+1} = (y_{n+1} + y_n)/2`.
        """
        y_mid = self._midpoint_state(next_ctx, y_curr_of_next)
        self._set_midpoint_time(next_ctx)
        return -(
            self._residual.mass_matrix().as_matrix()
            + 0.5 * next_ctx.deltat * self._residual.jacobian(y_mid)
        ).T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        y_mid = self._midpoint_state(ctx, final_fwd_sol)
        self._set_midpoint_time(ctx)
        op = self._residual.newton_jacobian(y_mid, 0.5 * ctx.deltat)
        return op.solve_transpose(-final_dqdu)


# =========================================================================
# HVP level: + HVP methods (same-step + cross-step)
# =========================================================================


class ImplicitMidpointHVP(
    HVPMixin[Array],
    ImplicitMidpointAdjoint[Array],
    Generic[Array],
):
    r"""Implicit midpoint with HVP capability for Hessian-vector products.

    The midpoint evaluation introduces chain-rule factors:

    - :math:`d^2R/dy_n^2`: factor :math:`(1/2)^2 = 1/4` from
      :math:`(dy_{\mathrm{mid}}/dy_n)^2`
    - :math:`d^2R/(dy_n\,dp)` and :math:`d^2R/(dp\,dy_n)`: factor
      :math:`1/2` from :math:`dy_{\mathrm{mid}}/dy_n`
    - :math:`d^2R/dp^2`: no chain-rule factor (param enters directly)

    Cross-step prev_* methods use ``next_ctx`` and ``y_curr_of_next``
    to compute the midpoint of step :math:`n+1`:
    :math:`y_{\mathrm{mid},n+1} = (y_{n+1} + y_n)/2`.
    """

    _residual: ImplicitODEResidualWithHVPProtocol[Array]

    def __init__(
        self, residual: ImplicitODEResidualWithHVPProtocol[Array]
    ) -> None:
        super().__init__(residual)

    # -- Same-step HVP methods --

    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t/4) \, (d^2f/dy^2)|_{y_{\mathrm{mid}}} \, w`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return (
            -0.25
            * ctx.deltat
            * self._hvp_residual.state_state_hvp(y_mid, adj_state, wvec)
        )

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t/2) \, (d^2f/dy\,dp)|_{y_{\mathrm{mid}}} \, v`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.state_param_hvp(y_mid, adj_state, vvec)
        )

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t/2) \, (d^2f/dp\,dy)|_{y_{\mathrm{mid}}} \, w`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.param_state_hvp(y_mid, adj_state, wvec)
        )

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`-\Delta t \, (d^2f/dp^2)|_{y_{\mathrm{mid}}} \, v`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return (
            -ctx.deltat
            * self._hvp_residual.param_param_hvp(y_mid, adj_state, vvec)
        )

    # -- Cross-step HVP methods --

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t_{n+1}/4)
        \, (d^2f/dy^2)|_{y_{\mathrm{mid},n+1}} \, w`."""
        y_mid = self._midpoint_state(next_ctx, y_curr_of_next)
        self._set_midpoint_time(next_ctx)
        return (
            -0.25
            * next_ctx.deltat
            * self._hvp_residual.state_state_hvp(y_mid, adj_state, wvec)
        )

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t_{n+1}/2)
        \, (d^2f/dy\,dp)|_{y_{\mathrm{mid},n+1}} \, v`."""
        y_mid = self._midpoint_state(next_ctx, y_curr_of_next)
        self._set_midpoint_time(next_ctx)
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.state_param_hvp(y_mid, adj_state, vvec)
        )

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`-(\Delta t_{n+1}/2)
        \, (d^2f/dp\,dy)|_{y_{\mathrm{mid},n+1}} \, w`."""
        y_mid = self._midpoint_state(next_ctx, y_curr_of_next)
        self._set_midpoint_time(next_ctx)
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.param_state_hvp(y_mid, adj_state, wvec)
        )

    # -- Mixed derivative HVP methods --
    # Nonzero because y_n and y_{n-1} both enter f through y_mid.

    def state_prev_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec_prev: Array,
    ) -> Array:
        r"""Compute :math:`\lambda^T \, \partial^2 R_n
        /(\partial y_n \, \partial y_{n-1}) \, w_{n-1}`."""
        y_mid = self._midpoint_state(ctx, y_curr)
        self._set_midpoint_time(ctx)
        return (
            -0.25
            * ctx.deltat
            * self._hvp_residual.state_state_hvp(y_mid, adj_state, wvec_prev)
        )

    def prev_state_curr_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec_curr_of_next: Array,
    ) -> Array:
        r"""Compute :math:`\lambda^T \, \partial^2 R_{n+1}
        /(\partial y_n \, \partial y_{n+1}) \, w_{n+1}`."""
        y_mid = self._midpoint_state(next_ctx, y_curr_of_next)
        self._set_midpoint_time(next_ctx)
        return (
            -0.25
            * next_ctx.deltat
            * self._hvp_residual.state_state_hvp(
                y_mid, adj_state, wvec_curr_of_next
            )
        )

