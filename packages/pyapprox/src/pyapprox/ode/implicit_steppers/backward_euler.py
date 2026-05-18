r"""Backward Euler time stepping residual with adjoint support.

The Backward Euler method is a first-order implicit time integrator:

.. math::

    M (y_n - y_{n-1}) - \Delta t \, f(y_n, t_n) = 0

Split into three classes via mixin composition:

- BackwardEulerStepper: core + sensitivity + quadrature + implicit
- BackwardEulerAdjoint: + adjoint methods
- BackwardEulerHVP: + HVP methods
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
from pyapprox.ode.mixins.quadrature import QuadratureMixin
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


class BackwardEulerStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
    ImplicitStepperMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Backward Euler time stepping residual (base level).

    First-order implicit method (A-stable):

    .. math::

        R(y_n) = M (y_n - y_{n-1}) - \Delta t \, f(y_n, t_n) = 0
    """

    _residual: ImplicitODEResidualProtocol[Array]

    def __init__(self, residual: ImplicitODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def _newton_coefficient(self) -> float:
        return self._ctx.deltat

    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._ctx.t_curr)
        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - self._ctx.deltat * self._residual(state)

    def jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dR/dy_n = M - \Delta t \, (df/dy)`."""
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
        return False

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1} = -M`.

        The :math:`f(y_n)` term does not depend on :math:`y_{n-1}`.
        """
        return -self._residual.mass_matrix().as_matrix()

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantRight,
        )
        return PiecewiseConstantRight


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class BackwardEulerAdjoint(
    AdjointMixin[Array],
    BackwardEulerStepper[Array],
    Generic[Array],
):
    """Backward Euler with adjoint capability for gradient computation."""

    _residual: ImplicitODEResidualWithParamJacobianProtocol[Array]

    def __init__(
        self, residual: ImplicitODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR/dp = -\Delta t \, (df/dp)|_{y_n, t_n}`."""
        self._residual.set_time(ctx.t_curr)
        return -ctx.deltat * self._adjoint_residual.param_jacobian(y_curr)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        r"""Compute :math:`(dR/dy_n)^T = (M - \Delta t \, J)^T`."""
        self._residual.set_time(ctx.t_curr)
        op = self._residual.newton_jacobian(y_curr, ctx.deltat)
        return TransposeLinearOperator(op)

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T = -M^T`."""
        return -self._residual.mass_matrix().as_matrix().T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        self._residual.set_time(ctx.t_curr)
        op = self._residual.newton_jacobian(
            final_fwd_sol, ctx.deltat
        )
        return op.solve_transpose(-final_dqdu)


# =========================================================================
# HVP level: + HVP methods (same-step + cross-step)
# =========================================================================


class BackwardEulerHVP(
    HVPMixin[Array],
    BackwardEulerAdjoint[Array],
    Generic[Array],
):
    r"""Backward Euler with HVP capability for Hessian-vector products.

    All same-step HVP methods are simple scalings of the underlying ODE
    residual HVPs evaluated at :math:`(y_n, t_n)`:

    .. math::

        \frac{d^2 R}{d(\cdot)^2} = -\Delta t \, \frac{d^2 f}{d(\cdot)^2}

    Cross-step prev_* methods return zero because R_{n+1} depends on y_n
    only through M(y_{n+1} - y_n), which is linear.
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
        r"""Compute :math:`(d^2R/dy_n^2) w = -\Delta t \, (d^2f/dy^2) w`."""
        self._residual.set_time(ctx.t_curr)
        return -ctx.deltat * self._hvp_residual.state_state_hvp(
            y_curr, adj_state, wvec
        )

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n \, dp) v = -\Delta t \, (d^2f/dy \, dp) v`."""
        self._residual.set_time(ctx.t_curr)
        return -ctx.deltat * self._hvp_residual.state_param_hvp(
            y_curr, adj_state, vvec
        )

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy_n) w = -\Delta t \, (d^2f/dp \, dy) w`."""
        self._residual.set_time(ctx.t_curr)
        return -ctx.deltat * self._hvp_residual.param_state_hvp(
            y_curr, adj_state, wvec
        )

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp^2) v = -\Delta t \, (d^2f/dp^2) v`."""
        self._residual.set_time(ctx.t_curr)
        return -ctx.deltat * self._hvp_residual.param_param_hvp(
            y_curr, adj_state, vvec
        )

    # -- Cross-step HVP methods (all zero for BE) --

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        return self._bkd.zeros(wvec.shape)

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        return self._bkd.zeros(adj_state.shape)

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        nparams = self._hvp_residual.nparams()
        return self._bkd.zeros((nparams,))
