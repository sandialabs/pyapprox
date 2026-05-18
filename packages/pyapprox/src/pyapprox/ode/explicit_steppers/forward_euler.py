"""Forward Euler time stepping residual with adjoint support.

The Forward Euler method is a first-order explicit time integrator:

    M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1}) = 0

Split into three classes via mixin composition:
- ForwardEulerStepper: core + sensitivity + quadrature
- ForwardEulerAdjoint: + adjoint methods
- ForwardEulerHVP: + HVP methods
"""

from typing import Generic

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    MassMatrixTransposeOperator,
)
from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin
from pyapprox.ode.mixins.quadrature import QuadratureMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array

# =========================================================================
# Base stepper: core + sensitivity + quadrature
# =========================================================================


class ForwardEulerStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Forward Euler time stepping residual (base level).

    First-order explicit time integrator:

    .. math::

        R(y_n) = M (y_n - y_{n-1}) - \Delta t \, f(y_{n-1}, t_{n-1}) = 0
    """

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._ctx.t_prev)
        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - self._ctx.deltat * self._residual(self._ctx.y_prev)

    def jacobian(self, state: Array) -> Array:
        return self._residual.mass_matrix().as_matrix()

    def linsolve(self, state: Array, residual: Array) -> Array:
        return self._residual.mass_matrix().solve(residual)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return True

    def is_one_step_solvable(self) -> bool:
        return True

    def has_prev_state_hessian(self) -> bool:
        return True

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1}` for forward sensitivity propagation.

        .. math::

            \frac{dR_n}{dy_{n-1}} = -(M + \Delta t \, J)

        where :math:`J = df/dy|_{y_{n-1}}`.
        """
        self._residual.set_time(ctx.t_prev)
        mass = self._residual.mass_matrix().as_matrix()
        jac = self._residual.jacobian(ctx.y_prev)
        return -mass - ctx.deltat * jac

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantLeft,
        )
        return PiecewiseConstantLeft


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class ForwardEulerAdjoint(
    AdjointMixin[Array],
    ForwardEulerStepper[Array],
    Generic[Array],
):
    """Forward Euler with adjoint capability for gradient computation."""

    def __init__(
        self, residual: ODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR/dp = -\Delta t \, (df/dp)|_{y_{n-1}}`."""
        self._adjoint_residual.set_time(ctx.t_prev)
        return -ctx.deltat * self._adjoint_residual.param_jacobian(ctx.y_prev)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        r"""Return :math:`(dR/dy_n)^T = M^T` (explicit, so Jacobian is mass matrix)."""
        return MassMatrixTransposeOperator(self._adjoint_residual.mass_matrix())

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T = -(\Delta t \, J + M)^T`."""
        self._adjoint_residual.set_time(next_ctx.t_prev)
        return -(
            next_ctx.deltat * self._adjoint_residual.jacobian(next_ctx.y_prev)
            + self._adjoint_residual.mass_matrix().as_matrix()
        ).T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        return -final_dqdu


# =========================================================================
# HVP level: + HVP methods (same-step + cross-step)
# =========================================================================


class ForwardEulerHVP(
    HVPMixin[Array],
    ForwardEulerAdjoint[Array],
    Generic[Array],
):
    r"""Forward Euler with HVP capability for Hessian-vector products.

    R_n = M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1})

    Same-step HVPs (d²R_n/dy_n²) are zero because R_n is linear in y_n.
    Cross-step HVPs (d²R_{k+1}/dy_k²) carry the nonlinear terms from
    f(y_k). param_param_hvp is nonzero because both terms depend on p.
    """

    def __init__(self, residual: ODEResidualWithHVPProtocol[Array]) -> None:
        super().__init__(residual)

    # -- Same-step HVP methods (all zero: R_n is linear in y_n) --

    def state_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_n/dy_n^2 = 0` (R_n linear in y_n)."""
        return self._bkd.zeros(y_curr.shape)

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_n/(dy_n \, dp) = 0` (R_n linear in y_n)."""
        return self._bkd.zeros(y_curr.shape)

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_n/(dp \, dy_n) = 0` (R_n linear in y_n)."""
        return self._bkd.zeros((self._hvp_residual.nparams(),))

    def param_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_n/dp^2 = -\Delta t \, (d^2f/dp^2)|_{y_{n-1}}`."""
        self._hvp_residual.set_time(ctx.t_prev)
        return -ctx.deltat * self._hvp_residual.param_param_hvp(
            ctx.y_prev, adj_state, vvec
        )

    # -- Cross-step HVP methods: d²R_{k+1}/d(y_k)² --

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_{k+1}/dy_k^2 = -\Delta t_{k+1} \, (d^2f/dy^2)|_{y_k}`."""
        self._hvp_residual.set_time(next_ctx.t_prev)
        return -next_ctx.deltat * self._hvp_residual.state_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_{k+1}/(dy_k \, dp) = -\Delta t_{k+1} \, (d^2f/(dy \, dp))|_{y_k}`."""
        self._hvp_residual.set_time(next_ctx.t_prev)
        return -next_ctx.deltat * self._hvp_residual.state_param_hvp(
            next_ctx.y_prev, adj_state, vvec
        )

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`d^2R_{k+1}/(dp \, dy_k) = -\Delta t_{k+1} \, (d^2f/(dp \, dy))|_{y_k}`."""
        self._hvp_residual.set_time(next_ctx.t_prev)
        return -next_ctx.deltat * self._hvp_residual.param_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )
