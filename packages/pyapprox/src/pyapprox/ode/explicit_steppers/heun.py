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

from typing import Generic, Union

from scipy.sparse import spmatrix

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
)
from pyapprox.ode.step_context import StepContext
from pyapprox.util.backends.protocols import Array, Backend


def _dense(bkd: Backend[Array], matrix: Union[spmatrix, Array]) -> Array:
    """Return the matrix as a dense backend array (no-op when dense)."""
    if isinstance(matrix, spmatrix):
        return bkd.asarray(matrix.toarray())
    return matrix


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

        # Stage: y_{n-1} + Δt·M^{-1}·k1. The ODE is M·dy/dt = f, so the
        # slope is M^{-1}f; for identity mass (collocation) the solve is
        # a no-op and this reduces to y_{n-1} + Δt·k1.
        mass = self._residual.mass_matrix()
        next_state = self._ctx.y_prev + self._ctx.deltat * mass.solve(k1)
        self._residual.set_time(self._ctx.t_curr)
        k2 = self._residual(next_state)

        return mass.apply(
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

    def is_multistage(self) -> bool:
        """Heun forms the predictor stage y + dt*M^{-1}*k1."""
        return True

    def has_prev_state_hessian(self) -> bool:
        return True

    def sensitivity_off_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1}` for forward sensitivity propagation.

        For Heun with :math:`k_1 = f(y_{n-1})`,
        :math:`k_2 = f(y_{n-1} + \Delta t \, M^{-1} k_1)`:

        .. math::

            \frac{dR_n}{dy_{n-1}} = -\left(M + \frac{\Delta t}{2}
            (J_1 + J_2 (I + \Delta t \, M^{-1} J_1))\right)

        One path for every mass: ``mass.solve`` is a passthrough for
        identity mass. :math:`M^{-1} J_1` is inherently dense (inverse
        fill-in), so sparse operands are normalized to dense before
        mixing with it — a no-op for backend arrays.
        """
        mass_obj = self._residual.mass_matrix()

        self._residual.set_time(ctx.t_prev)
        k1_jac = self._residual.jacobian(ctx.y_prev)

        k1 = self._residual(ctx.y_prev)
        k2_state = ctx.y_prev + ctx.deltat * mass_obj.solve(k1)

        self._residual.set_time(ctx.t_curr)
        k2_jac = self._residual.jacobian(k2_state)

        minv_j1 = mass_obj.solve(_dense(self._bkd, k1_jac))
        inner = (
            _dense(self._bkd, k1_jac)
            + _dense(self._bkd, k2_jac)
            + ctx.deltat * (k2_jac @ minv_j1)
        )
        return -(
            _dense(self._bkd, mass_obj.as_matrix())
            + 0.5 * ctx.deltat * inner
        )

    # -- QuadratureMixin --

    def quadrature_samples_weights(self, times: Array) -> tuple[Array, Array]:
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
        self, residual: ODEResidualProtocol[Array]
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
        + \partial f/\partial y|_z \cdot \Delta t \, M^{-1} \, dk_1/dp`
        with the stage :math:`z = y_{n-1} + \Delta t \, M^{-1} k_1`
        (the ODE is :math:`M \, dy/dt = f`, so the slope is
        :math:`M^{-1} f`; both solves are passthroughs for identity
        mass).
        """
        mass_obj = self._residual.mass_matrix()

        # k1 stage
        self._residual.set_time(ctx.t_prev)
        k1_param_jac = self._adjoint_residual.param_jacobian(ctx.y_prev)

        # k2 stage: z = y_{n-1} + Δt·M^{-1}·k1
        k1 = self._residual(ctx.y_prev)
        k2_state = ctx.y_prev + ctx.deltat * mass_obj.solve(k1)

        self._residual.set_time(ctx.t_curr)
        k2_state_jac = self._residual.jacobian(k2_state)
        k2_param_jac = self._adjoint_residual.param_jacobian(k2_state)

        # Chain rule: dk2/dp = ∂f/∂p + ∂f/∂y · Δt · M^{-1} · dk1/dp
        minv_k1_param_jac = mass_obj.solve(
            _dense(self._bkd, k1_param_jac)
        )
        return -(
            0.5
            * ctx.deltat
            * (
                k1_param_jac
                + k2_param_jac
                + ctx.deltat * (k2_state_jac @ minv_k1_param_jac)
            )
        )

    def adjoint_diag_jacobian(
        self, ctx: StepContext[Array], y_curr: Array
    ) -> LinearOperatorProtocol[Array]:
        return MassMatrixTransposeOperator(self._adjoint_residual.mass_matrix())

    def adjoint_off_diag_jacobian(
        self, next_ctx: StepContext[Array], y_curr_of_next: Array
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T` for adjoint coupling.

        The transpose of the forward sensitivity off-diagonal block —
        delegated so the two-stage chain rule (with its
        :math:`M^{-1}` stage slope) has a single source of truth.
        """
        return self.sensitivity_off_diag_jacobian(
            next_ctx, y_curr_of_next
        ).T

    def adjoint_initial_condition(
        self, ctx: StepContext[Array], final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`M^T \lambda_N = -dQ/dy_N` (explicit:
        :math:`dR_N/dy_N = M`)."""
        return self._adjoint_residual.mass_matrix().solve_transpose(
            -final_dqdu
        )


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

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
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
        r"""Compute :math:`(d^2R / dp^2) v` contracted with adjoint.

        The stage is :math:`z = y_{n-1} + \Delta t \, M^{-1} k_1`, so
        :math:`dz/dp = \Delta t \, M^{-1} \, dk_1/dp` and every stage
        pullback carries :math:`M^{-T}` (passthroughs for identity
        mass).
        """
        dt = ctx.deltat
        mass_obj = self._residual.mass_matrix()

        # Stage 1
        self._residual.set_time(ctx.t_prev)
        k1 = self._residual(ctx.y_prev)
        dk1_dp = self._adjoint_residual.param_jacobian(ctx.y_prev)

        k1_pp_hvp = self._hvp_residual.param_param_hvp(ctx.y_prev, adj_state, vvec)

        # Stage 2
        z = ctx.y_prev + dt * mass_obj.solve(k1)
        self._residual.set_time(ctx.t_curr)
        J2 = self._residual.jacobian(z)

        dz_dp_v = dt * mass_obj.solve(
            self._bkd.flatten(_dense(self._bkd, dk1_dp) @ vvec)
        )
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: d2f/dp2 at z, direction v
        k2_term1 = self._hvp_residual.param_param_hvp(z, adj_state, vvec)

        # Term 2: d2f/(dp dz) at z, direction dz/dp v
        k2_term2 = self._hvp_residual.param_state_hvp(z, adj_state, dz_dp_v_flat)

        # Term 3: (dz/dp)^T f_zz(z) (dz/dp v) = dt dk1_dp^T M^-T h
        h_dz_dp_v = self._hvp_residual.state_state_hvp(z, adj_state, dz_dp_v_flat)
        h_dz_dp_v_flat = mass_obj.solve_transpose(
            self._bkd.flatten(h_dz_dp_v)
        )
        k2_term3 = dt * (dk1_dp.T @ h_dz_dp_v_flat)

        # Term 4: adj^T J_z d2z/dp2 v = dt f_pp(y; M^-T J_z^T adj) v
        J2_T_adj = mass_obj.solve_transpose(J2.T @ adj_state)
        self._residual.set_time(ctx.t_prev)
        k2_term4 = dt * self._hvp_residual.param_param_hvp(ctx.y_prev, J2_T_adj, vvec)

        # Term 5: (dz/dp)^T f_zp(z) v = dt dk1_dp^T M^-T sp
        self._residual.set_time(ctx.t_curr)
        sp_hvp = self._hvp_residual.state_param_hvp(z, adj_state, vvec)
        k2_term5 = dt * (
            dk1_dp.T
            @ self._bkd.reshape(
                mass_obj.solve_transpose(self._bkd.flatten(sp_hvp)),
                (-1, 1),
            )
        )

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
        r"""Compute :math:`(d^2R_{k+1}/dy_k^2) w` contracted with adjoint.

        With the stage :math:`z = y_k + \Delta t \, M^{-1} f(y_k)` the
        stage sensitivity is :math:`S = I + \Delta t \, M^{-1} J_1`,
        so the stage-2 Hessian is pulled back through :math:`S^T` and
        the stage-curvature weight carries :math:`M^{-T}`.
        """
        dt = next_ctx.deltat
        mass_obj = self._residual.mass_matrix()

        # Stage 1: k1 = f(y_k)
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)

        k1_ss_hvp = self._hvp_residual.state_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )

        # Stage 2: k2 = f(z) where z = y_k + dt*M^-1*k1
        z = next_ctx.y_prev + dt * mass_obj.solve(k1)
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        # S = dz/dy = I + dt*M^-1*J1
        nstates = next_ctx.y_prev.shape[0]
        stage_sens = self._bkd.eye(nstates) + dt * mass_obj.solve(
            _dense(self._bkd, J1)
        )

        # Term 1: S^T f_zz(z; adj) (S w)
        scaled_wvec = stage_sens @ wvec
        h2_scaled = self._hvp_residual.state_state_hvp(
            z, adj_state, scaled_wvec
        )
        h2_scaled_flat = self._bkd.flatten(h2_scaled)
        k2_term1 = stage_sens.T @ h2_scaled_flat

        # Term 2: adj^T J2 dS/dy w = dt f_yy(y_k; M^-T J2^T adj) w
        J2_T_adj = mass_obj.solve_transpose(J2.T @ adj_state)
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
        r"""Compute :math:`(d^2R_{k+1} / dy_k \, dp) v` contracted with adjoint.

        Stage pullbacks use :math:`S = I + \Delta t \, M^{-1} J_1` and
        :math:`dz/dp = \Delta t \, M^{-1} \, dk_1/dp`; the
        stage-curvature weight carries :math:`M^{-T}`.
        """
        dt = next_ctx.deltat
        mass_obj = self._residual.mass_matrix()

        # Stage 1
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)
        dk1_dp = self._adjoint_residual.param_jacobian(next_ctx.y_prev)

        k1_sp_hvp = self._hvp_residual.state_param_hvp(
            next_ctx.y_prev, adj_state, vvec
        )

        # Stage 2
        z = next_ctx.y_prev + dt * mass_obj.solve(k1)
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        nstates = next_ctx.y_prev.shape[0]
        stage_sens = self._bkd.eye(nstates) + dt * mass_obj.solve(
            _dense(self._bkd, J1)
        )
        dz_dp_v = dt * mass_obj.solve(
            self._bkd.flatten(_dense(self._bkd, dk1_dp) @ vvec)
        )
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: S^T f_zp(z; adj) v
        sp_hvp_z = self._hvp_residual.state_param_hvp(z, adj_state, vvec)
        k2_term1 = stage_sens.T @ self._bkd.flatten(sp_hvp_z)

        # Term 2: S^T f_zz(z; adj) (dz/dp v)
        ss_hvp_z = self._hvp_residual.state_state_hvp(
            z, adj_state, dz_dp_v_flat
        )
        k2_term2 = stage_sens.T @ self._bkd.flatten(ss_hvp_z)

        # Term 3: adj^T J2 dS/dp v = dt f_yp(y_k; M^-T J2^T adj) v
        J2_T_adj = mass_obj.solve_transpose(J2.T @ adj_state)
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
        r"""Compute :math:`(d^2R_{k+1} / dp \, dy_k) w` contracted with adjoint.

        Stage pullbacks use :math:`S = I + \Delta t \, M^{-1} J_1` and
        :math:`(dz/dp)^T = \Delta t \, (dk_1/dp)^T M^{-T}`; the
        stage-curvature weight carries :math:`M^{-T}`.
        """
        dt = next_ctx.deltat
        mass_obj = self._residual.mass_matrix()

        # Stage 1
        self._residual.set_time(next_ctx.t_prev)
        k1 = self._residual(next_ctx.y_prev)
        J1 = self._residual.jacobian(next_ctx.y_prev)
        dk1_dp = self._adjoint_residual.param_jacobian(next_ctx.y_prev)

        k1_ps_hvp = self._hvp_residual.param_state_hvp(
            next_ctx.y_prev, adj_state, wvec
        )

        # Stage 2
        z = next_ctx.y_prev + dt * mass_obj.solve(k1)
        self._residual.set_time(next_ctx.t_curr)
        J2 = self._residual.jacobian(z)

        nstates = next_ctx.y_prev.shape[0]
        stage_sens = self._bkd.eye(nstates) + dt * mass_obj.solve(
            _dense(self._bkd, J1)
        )
        dz_dy_w = stage_sens @ wvec

        # Term 1: f_pz(z; adj) (S w)
        k2_term1 = self._hvp_residual.param_state_hvp(
            z, adj_state, dz_dy_w
        )

        # Term 2: (dz/dp)^T f_zz(z; adj) (S w) = dt dk1_dp^T M^-T H
        H_z_dz_dy_w = self._hvp_residual.state_state_hvp(
            z, adj_state, dz_dy_w
        )
        k2_term2 = dt * (
            dk1_dp.T
            @ self._bkd.reshape(
                mass_obj.solve_transpose(
                    self._bkd.flatten(H_z_dz_dy_w)
                ),
                (-1, 1),
            )
        )

        # Term 3: adj^T J2 d/dp[dS w] = dt f_py(y_k; M^-T J2^T adj) w
        J2_T_adj = mass_obj.solve_transpose(J2.T @ adj_state)
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
