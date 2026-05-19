"""Heun's method (RK2) time stepping residual with adjoint support.

Heun's method is a second-order explicit Runge-Kutta method:

    k1 = f(y_{n-1}, t_{n-1})
    k2 = f(y_{n-1} + Δt·k1, t_n)
    M·(y_n - y_{n-1}) = (Δt/2)·(k1 + k2)

Split into three classes via mixin composition:
- HeunStepper: core + sensitivity + quadrature
- HeunAdjoint: + adjoint methods
- HeunHVP: + HVP methods
"""

from typing import Generic

from pyapprox.ode.linear_operator import (
    LinearOperatorProtocol,
    MassMatrixTransposeOperator,
)
from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin
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


class HeunStepper(
    SensitivityMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Heun's method (RK2) time stepping residual (base level).

    Two-stage explicit Runge-Kutta method (2nd order):

    .. math::

        k_1 = f(y_{n-1}, t_{n-1})

        k_2 = f(y_{n-1} + \Delta t \cdot k_1, t_n)

        R(y_n) = M (y_n - y_{n-1}) - \frac{\Delta t}{2} (k_1 + k_2) = 0
    """

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def __call__(self, state: Array) -> Array:
        # k1 = f(y_{n-1}, t_{n-1})
        self._residual.set_time(self._ctx.t_prev)
        k1 = self._residual(self._ctx.y_prev)

        # k2 = f(y_{n-1} + Δt·k1, t_n)
        next_state = self._ctx.y_prev + self._ctx.deltat * k1
        self._residual.set_time(self._ctx.t_curr)
        k2 = self._residual(next_state)

        return self._residual.mass_matrix().apply(
            state - self._ctx.y_prev
        ) - 0.5 * self._ctx.deltat * (k1 + k2)

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

        For Heun with :math:`k_1 = f(y_{n-1})`,
        :math:`k_2 = f(y_{n-1} + \Delta t \cdot k_1)`:

        .. math::

            \frac{dR_n}{dy_{n-1}} = -\left(M + \frac{\Delta t}{2}
            (J_1 + J_2 (M + \Delta t \, J_1))\right)
        """
        self._residual.set_time(ctx.t_prev)
        k1_jac = self._residual.jacobian(ctx.y_prev)

        k1 = self._residual(ctx.y_prev)
        k2_state = ctx.y_prev + ctx.deltat * k1

        self._residual.set_time(ctx.t_curr)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix().as_matrix()

        # dR/dy_{n-1} = -(M + (Δt/2)·(J1 + J2·(M + Δt·J1)))
        inner = k1_jac + k2_jac @ (mass + ctx.deltat * k1_jac)
        return -(mass + 0.5 * ctx.deltat * inner)

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


class HeunAdjoint(
    AdjointMixin[Array],
    HeunStepper[Array],
    Generic[Array],
):
    """Heun's method with adjoint capability for gradient computation."""

    def __init__(
        self, residual: ODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute the parameter Jacobian :math:`dR/dp` for one time step.

        .. math::

            \frac{dR}{dp} = -\frac{\Delta t}{2}
            \left(\frac{dk_1}{dp} + \frac{dk_2}{dp}\right)

        where :math:`dk_1/dp = \partial f/\partial p|_{y_{n-1}}` and
        :math:`dk_2/dp = \partial f/\partial p|_z
        + \partial f/\partial y|_z \cdot \Delta t \cdot dk_1/dp`
        with :math:`z = y_{n-1} + \Delta t \cdot k_1`.
        """
        # k1 stage
        self._residual.set_time(ctx.t_prev)
        k1_param_jac = self._adjoint_residual.param_jacobian(ctx.y_prev)

        # k2 stage: k2_state = y_{n-1} + Δt·k1
        k1 = self._residual(ctx.y_prev)
        k2_state = ctx.y_prev + ctx.deltat * k1

        self._residual.set_time(ctx.t_curr)
        k2_state_jac = self._residual.jacobian(k2_state)
        k2_param_jac = self._adjoint_residual.param_jacobian(k2_state)

        # Chain rule: dk2/dp = ∂f/∂p + ∂f/∂y · Δt · dk1/dp
        return -(
            0.5
            * ctx.deltat
            * (
                k1_param_jac
                + k2_param_jac
                + ctx.deltat * (k2_state_jac @ k1_param_jac)
            )
        )

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        return MassMatrixTransposeOperator(self._adjoint_residual.mass_matrix())

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute the off-diagonal Jacobian for adjoint coupling.

        For Heun, :math:`dR_{n+1}/dy_n` involves derivatives through both
        :math:`k_1` and :math:`k_2` stages:

        .. math::

            \frac{dR_{n+1}}{dy_n} = -\left(M + \frac{\Delta t}{2}
            (J_1 + J_2 (M + \Delta t \, J_1))\right)

        Returns the transpose :math:`(dR_{n+1}/dy_n)^T`.
        """
        self._residual.set_time(next_ctx.t_prev)
        k1_jac = self._residual.jacobian(next_ctx.y_prev)

        k1 = self._residual(next_ctx.y_prev)
        k2_state = next_ctx.y_prev + next_ctx.deltat * k1

        self._residual.set_time(next_ctx.t_curr)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix().as_matrix()

        inner = k1_jac + k2_jac @ (
            mass + next_ctx.deltat * k1_jac
        )
        jac = -(mass + 0.5 * next_ctx.deltat * inner)
        return jac.T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        return -final_dqdu


# =========================================================================
# HVP level: + HVP methods (same-step + cross-step)
# =========================================================================


class HeunHVP(
    HVPMixin[Array],
    HeunAdjoint[Array],
    Generic[Array],
):
    """Heun's method with HVP capability for Hessian-vector products.

    Heun evaluates f at two stages (t_prev, t_curr). Within each HVP method,
    sub-calls to self._hvp_residual.*_hvp(...) inherit the most recent
    set_time(). When a stage-1 sub-call follows a stage-2 J2 evaluation
    (e.g. for the J2^T · adj contraction term), reset the residual time to
    next_ctx.t_prev (or ctx.t_prev) IMMEDIATELY before the sub-call. Failing
    to do so silently uses the stage-2 time on a stage-1 quantity — invisible
    under autonomous f, wrong otherwise.
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
        r"""Compute :math:`(d^2R / dp^2) v` contracted with adjoint."""
        dt = ctx.deltat

        # Stage 1
        self._residual.set_time(ctx.t_prev)
        k1 = self._residual(ctx.y_prev)
        dk1_dp = self._adjoint_residual.param_jacobian(ctx.y_prev)

        k1_pp_hvp = self._hvp_residual.param_param_hvp(ctx.y_prev, adj_state, vvec)

        # Stage 2
        z = ctx.y_prev + dt * k1
        self._residual.set_time(ctx.t_curr)
        J2 = self._residual.jacobian(z)

        dz_dp_v = dt * (dk1_dp @ vvec)
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: ∂²f/∂p²|_z · v
        k2_term1 = self._hvp_residual.param_param_hvp(z, adj_state, vvec)

        # Term 2: (∂²f/∂p∂z|_z) · dz/dp · v
        k2_term2 = self._hvp_residual.param_state_hvp(z, adj_state, dz_dp_v_flat)

        # Term 3: H_z · (dz/dp)² contribution
        h_dz_dp_v = self._hvp_residual.state_state_hvp(z, adj_state, dz_dp_v_flat)
        h_dz_dp_v_flat = self._bkd.flatten(h_dz_dp_v)
        k2_term3 = dt * (dk1_dp.T @ h_dz_dp_v_flat)

        # Term 4: J_z · dt · ∂²f/∂p²|_y · v
        J2_T_adj = J2.T @ adj_state
        self._residual.set_time(ctx.t_prev)
        k2_term4 = dt * self._hvp_residual.param_param_hvp(ctx.y_prev, J2_T_adj, vvec)

        # Term 5: ∂J_z/∂p · v · dt · dk1_dp
        self._residual.set_time(ctx.t_curr)
        sp_hvp = self._hvp_residual.state_param_hvp(z, adj_state, vvec)
        k2_term5 = dt * (dk1_dp.T @ self._bkd.reshape(sp_hvp, (-1, 1)))

        result = (
            self._bkd.flatten(k1_pp_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
            + self._bkd.flatten(k2_term4)
            + self._bkd.flatten(k2_term5)
        )
        return -0.5 * dt * result

    # -- Cross-step HVP methods: d²R_{k+1}/d(y_k)² --

    def prev_state_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R_{k+1}/dy_k^2) w` contracted with adjoint."""
        dt = next_ctx.deltat

        # Stage 1: k1 = f(y_k)
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)
        mass = self._residual.mass_matrix().as_matrix()

        k1_ss_hvp = self._hvp_residual.state_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )

        # Stage 2: k2 = f(z) where z = y_k + dt*k1
        z = next_ctx.y_prev + dt * k1
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1

        # Term 1: H2 · (dz/dy · w) weighted by dz/dy
        scaled_wvec = dz_dy @ wvec
        h2_scaled = self._hvp_residual.state_state_hvp(
            z, adj_state, scaled_wvec
        )
        h2_scaled_flat = self._bkd.flatten(h2_scaled)
        k2_term1 = dz_dy.T @ h2_scaled_flat

        # Term 2: J2 · dt · H1 · w (with adjoint = J2^T · adj)
        J2_T_adj = J2.T @ adj_state
        self._residual.set_time(next_ctx.t_prev)
        k2_term2 = dt * self._hvp_residual.state_state_hvp(
            next_ctx.y_prev, J2_T_adj, wvec
        )

        result = (
            self._bkd.flatten(k1_ss_hvp)
            + k2_term1
            + self._bkd.flatten(k2_term2)
        )
        return -0.5 * dt * result

    def prev_state_param_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R_{k+1} / dy_k \, dp) v` contracted with adjoint."""
        dt = next_ctx.deltat

        # Stage 1
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)
        mass = self._residual.mass_matrix().as_matrix()
        dk1_dp = self._adjoint_residual.param_jacobian(next_ctx.y_prev)

        k1_sp_hvp = self._hvp_residual.state_param_hvp(
            next_ctx.y_prev, adj_state, vvec
        )

        # Stage 2
        z = next_ctx.y_prev + dt * k1
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1
        dz_dp_v = dt * (dk1_dp @ vvec)
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: dz/dy^T · state_param_hvp(z, adj, v)
        sp_hvp_z = self._hvp_residual.state_param_hvp(z, adj_state, vvec)
        k2_term1 = dz_dy.T @ self._bkd.flatten(sp_hvp_z)

        # Term 2: dz/dy^T · state_state_hvp(z, adj, dz/dp · v)
        ss_hvp_z = self._hvp_residual.state_state_hvp(
            z, adj_state, dz_dp_v_flat
        )
        k2_term2 = dz_dy.T @ self._bkd.flatten(ss_hvp_z)

        # Term 3: (J_z^T · adj)^T · dt · state_param_hvp at y_k
        J2_T_adj = J2.T @ adj_state
        self._residual.set_time(next_ctx.t_prev)
        k2_term3 = dt * self._hvp_residual.state_param_hvp(
            next_ctx.y_prev, J2_T_adj, vvec
        )

        result = (
            self._bkd.flatten(k1_sp_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result

    def prev_param_state_hvp(
        self,
        next_ctx: StepContext[Array],
        y_curr_of_next: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R_{k+1} / dp \, dy_k) w` contracted with adjoint."""
        dt = next_ctx.deltat

        # Stage 1
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)
        mass = self._residual.mass_matrix().as_matrix()
        dk1_dp = self._adjoint_residual.param_jacobian(next_ctx.y_prev)

        k1_ps_hvp = self._hvp_residual.param_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )

        # Stage 2
        z = next_ctx.y_prev + dt * k1
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1
        dz_dy_w = dz_dy @ wvec

        # Term 1: param_state_hvp(z, adj, dz/dy · w)
        k2_term1 = self._hvp_residual.param_state_hvp(
            z, adj_state, dz_dy_w
        )

        # Term 2: H_z · (dz/dy · w) · dt · df/dp|_y
        H_z_dz_dy_w = self._hvp_residual.state_state_hvp(
            z, adj_state, dz_dy_w
        )
        k2_term2 = dt * (
            dk1_dp.T @ self._bkd.reshape(H_z_dz_dy_w, (-1, 1))
        )

        # Term 3: (J_z^T · adj) · dt · param_state_hvp at y_k
        J2_T_adj = J2.T @ adj_state
        self._residual.set_time(next_ctx.t_prev)
        k2_term3 = dt * self._hvp_residual.param_state_hvp(
            next_ctx.y_prev, J2_T_adj, wvec
        )

        result = (
            self._bkd.flatten(k1_ps_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result
