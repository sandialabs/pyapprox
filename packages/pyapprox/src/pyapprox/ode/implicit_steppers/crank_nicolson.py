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

from functools import partial
from typing import Generic, Optional, TypeVar

from pyapprox.ode.implicit_stage_solve import (
    ImplicitStageSolve,
    newton_matrix,
)
from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    TransposeLinearOperator,
)
from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin
from pyapprox.ode.mixins.implicit import ImplicitStepperMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin
from pyapprox.ode.off_diagonal import MassPlusJacobianOffDiagonal
from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualWithHVPProtocol,
    ImplicitODEResidualWithParamJacobianProtocol,
    ODEResidualProtocol,
)
from pyapprox.ode.protocols.residual_form_system import ResidualFormSystemProtocol
from pyapprox.ode.step_context import StepContext
from pyapprox.ode.time_quadrature import (
    OneStepMassTelescope,
    StepResidualDeclaration,
    TrajectoryQuadratureProtocol,
    trapezoidal_quadrature,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import ActionOperatorProtocol, FormatProtocol
from pyapprox.util.linalg.operators.solvers import require_transposes

F = TypeVar("F")

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

    def __init__(
        self,
        residual: ODEResidualProtocol[Array],
        system: Optional[ResidualFormSystemProtocol[Array, Array, F]] = None,
    ) -> None:
        super().__init__(residual)
        self._stage = ImplicitStageSolve(residual, system)

    def _newton_operator(
        self, ctx: StepContext[Array], state: Array
    ) -> LinearOperatorProtocol[Array]:
        r""":math:`dR/dy_n = M - (\Delta t/2) \, (df/dy)(y_n, t_n)`, the implicit
        system's shifted Jacobian at :math:`y_n` with shift :math:`2/\Delta t`,
        scaled by :math:`\Delta t/2`.

        The rate passed is the backward difference
        :math:`(y_n - y_{n-1})/\Delta t`, not the trapezoidal stage rate;
        mass-matrix systems do not read it, a system whose shifted
        Jacobian depends on :math:`\dot y` would need the stage rate.
        """
        return self._stage.newton_operator(
            state,
            (state - ctx.y_prev) / ctx.deltat,
            ctx.t_curr,
            0.5 * ctx.deltat,
            self._stage.stage_point(ctx.t_curr),
            partial(
                newton_matrix, self._residual, state, ctx.t_curr, 0.5 * ctx.deltat
            ),
        )

    def __call__(self, state: Array) -> Array:
        current_res = self._residual(self._ctx.y_prev, self._ctx.t_prev)
        next_res = self._residual(state, self._ctx.t_curr)
        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - 0.5 * self._ctx.deltat * (current_res + next_res)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return False

    def is_one_step_solvable(self) -> bool:
        return False

    def has_prev_state_hessian(self) -> bool:
        return True

    def off_diag_operator(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> ActionOperatorProtocol[Array, FormatProtocol]:
        r""":math:`dR_n/dy_{n-1} = -(M + (\Delta t/2)\,J_{n-1})` as an
        operator; its transpose is the adjoint's off-diagonal block."""
        return MassPlusJacobianOffDiagonal(
            self._residual.mass_matrix(),
            self._residual.jacobian(ctx.y_prev, ctx.t_prev),
            0.5 * ctx.deltat,
            y_curr.shape[0],
            self._bkd,
        )

    # -- QuadratureMixin --

    def trajectory_quadrature(
        self, times: Array
    ) -> TrajectoryQuadratureProtocol[Array]:
        """Trapezoidal rule — Crank-Nicolson IS the trapezoid scheme.

        Its samples are the states and times where the step evaluates
        ``F``, so it declares the one-step mass telescope.
        """
        return trapezoidal_quadrature(
            times,
            self._bkd,
            StepResidualDeclaration(
                OneStepMassTelescope(), self._residual.mass_matrix()
            ),
        )


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
        self,
        residual: ODEResidualProtocol[Array],
        system: Optional[ResidualFormSystemProtocol[Array, Array, F]] = None,
    ) -> None:
        super().__init__(residual, system)
        if system is not None:
            require_transposes(system.solver(), type(self).__name__)

    def _param_jacobian_impl(self, ctx: StepContext[Array], y_curr: Array) -> Array:
        r"""Compute :math:`dR/dp`.

        .. math::

            \frac{dR}{dp} = -\frac{\Delta t}{2}
            \Bigl[\frac{df}{dp}\Big|_{y_{n-1}, t_{n-1}}
            + \frac{df}{dp}\Big|_{y_n, t_n}\Bigr]
        """
        current_param_jac = self._adjoint_residual.param_jacobian(
            ctx.y_prev, ctx.t_prev
        )
        next_param_jac = self._adjoint_residual.param_jacobian(y_curr, ctx.t_curr)
        return -0.5 * ctx.deltat * (current_param_jac + next_param_jac)

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        r"""Compute :math:`(dR/dy_n)^T = (M - (\Delta t/2) \, J)^T`."""
        return TransposeLinearOperator(self._newton_operator(ctx, y_curr))

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        return self._newton_operator(ctx, final_fwd_sol).solve_transpose(-final_dqdu)


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

    def __init__(
        self,
        residual: ODEResidualProtocol[Array],
        system: Optional[ResidualFormSystemProtocol[Array, Array, F]] = None,
    ) -> None:
        super().__init__(residual, system)

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
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.state_state_hvp(y_curr, ctx.t_curr, adj_state, wvec)
        )

    def state_param_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n \, dp) v`."""
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.state_param_hvp(y_curr, ctx.t_curr, adj_state, vvec)
        )

    def param_state_hvp(
        self,
        ctx: StepContext[Array],
        y_curr: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy_n) w`."""
        return (
            -0.5
            * ctx.deltat
            * self._hvp_residual.param_state_hvp(y_curr, ctx.t_curr, adj_state, wvec)
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
        hvp_nm1 = self._hvp_residual.param_param_hvp(
            ctx.y_prev, ctx.t_prev, adj_state, vvec
        )

        # Contribution from y_n term
        hvp_n = self._hvp_residual.param_param_hvp(y_curr, ctx.t_curr, adj_state, vvec)

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
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.state_state_hvp(
                next_ctx.y_prev, next_ctx.t_prev, adj_state, wvec
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
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.state_param_hvp(
                next_ctx.y_prev, next_ctx.t_prev, adj_state, vvec
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
        return (
            -0.5
            * next_ctx.deltat
            * self._hvp_residual.param_state_hvp(
                next_ctx.y_prev, next_ctx.t_prev, adj_state, wvec
            )
        )
