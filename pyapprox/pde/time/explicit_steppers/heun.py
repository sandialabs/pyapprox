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

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.mixins.adjoint import AdjointMixin
from pyapprox.pde.time.mixins.core import CoreStepperMixin
from pyapprox.pde.time.mixins.hvp import HVPMixin
from pyapprox.pde.time.mixins.quadrature import QuadratureMixin
from pyapprox.pde.time.mixins.sensitivity import SensitivityMixin
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array


# =========================================================================
# Base stepper: core + sensitivity + quadrature
# =========================================================================


class HeunStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
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
        self._residual.set_time(self._time)
        k1 = self._residual(self._prev_state)

        # k2 = f(y_{n-1} + Δt·k1, t_n)
        next_state = self._prev_state + self._deltat * k1
        self._residual.set_time(self._time + self._deltat)
        k2 = self._residual(next_state)

        return self._residual.apply_mass_matrix(
            state - self._prev_state
        ) - 0.5 * self._deltat * (k1 + k2)

    def jacobian(self, state: Array) -> Array:
        return self._residual.mass_matrix(state.shape[0])

    def linsolve(self, state: Array, residual: Array) -> Array:
        return solve_maybe_sparse(self._bkd, self.jacobian(state), residual)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return True

    def has_prev_state_hessian(self) -> bool:
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1}` for forward sensitivity propagation.

        For Heun with :math:`k_1 = f(y_{n-1})`,
        :math:`k_2 = f(y_{n-1} + \Delta t \cdot k_1)`:

        .. math::

            \frac{dR_n}{dy_{n-1}} = -\left(M + \frac{\Delta t}{2}
            (J_1 + J_2 (M + \Delta t \, J_1))\right)
        """
        self._residual.set_time(self._time)
        k1_jac = self._residual.jacobian(fsol_nm1)

        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + deltat * k1

        self._residual.set_time(self._time + deltat)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix(fsol_nm1.shape[0])

        # dR/dy_{n-1} = -(M + (Δt/2)·(J1 + J2·(M + Δt·J1)))
        return -(mass + 0.5 * deltat * (k1_jac + k2_jac @ (mass + deltat * k1_jac)))

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseLinear,
        )
        return PiecewiseLinear


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

    def _param_jacobian_impl(self, fsol_nm1: Array, fsol_n: Array) -> Array:
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
        self._residual.set_time(self._time)
        k1_param_jac = self._adjoint_residual.param_jacobian(fsol_nm1)

        # k2 stage: k2_state = y_{n-1} + Δt·k1
        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + self._deltat * k1

        self._residual.set_time(self._time + self._deltat)
        k2_state_jac = self._residual.jacobian(k2_state)
        k2_param_jac = self._adjoint_residual.param_jacobian(k2_state)

        # Chain rule: dk2/dp = ∂f/∂p + ∂f/∂y · Δt · dk1/dp
        return -(
            0.5
            * self._deltat
            * (
                k1_param_jac
                + k2_param_jac
                + self._deltat * (k2_state_jac @ k1_param_jac)
            )
        )

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        return self._adjoint_residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        r"""Compute the off-diagonal Jacobian for adjoint coupling.

        For Heun, :math:`dR_{n+1}/dy_n` involves derivatives through both
        :math:`k_1` and :math:`k_2` stages:

        .. math::

            \frac{dR_{n+1}}{dy_n} = -\left(M + \frac{\Delta t}{2}
            (J_1 + J_2 (M + \Delta t \, J_1))\right)

        Returns the transpose :math:`(dR_{n+1}/dy_n)^T`.
        """
        self._residual.set_time(self._time)
        k1_jac = self._residual.jacobian(fsol_n)

        k1 = self._residual(fsol_n)
        k2_state = fsol_n + deltat_np1 * k1

        self._residual.set_time(self._time + deltat_np1)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix(fsol_n.shape[0])

        jac = -(
            mass + 0.5 * deltat_np1 * (k1_jac + k2_jac @ (mass + deltat_np1 * k1_jac))
        )
        return jac.T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        return -final_dqdu


# =========================================================================
# HVP level: + four HVP methods
# =========================================================================


class HeunHVP(
    HVPMixin[Array],
    HeunAdjoint[Array],
    Generic[Array],
):
    """Heun's method with HVP capability for Hessian-vector products."""

    def __init__(self, residual: ODEResidualWithHVPProtocol[Array]) -> None:
        super().__init__(residual)

    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_{n-1}^2) w` contracted with adjoint.

        With :math:`R = y_n - y_{n-1} - (\Delta t/2)(k_1 + k_2)`,
        :math:`k_1 = f(y_{n-1})`, :math:`k_2 = f(z)` where
        :math:`z = y_{n-1} + \Delta t \, k_1`:

        .. math::

            \frac{d^2 R}{dy^2} = -\frac{\Delta t}{2}
            \left(H_1 + H_2 (I + \Delta t \, J_1)^2
            + J_2 \, \Delta t \, H_1\right)

        where :math:`H_1, H_2 = d^2f/dy^2` at :math:`y_{n-1}` and :math:`z`.
        """
        dt = self._deltat

        # Stage 1: k1 = f(y_{n-1})
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])

        k1_ss_hvp = self._hvp_residual.state_state_hvp(fsol_nm1, adj_state, wvec)

        # Stage 2: k2 = f(z) where z = y_{n-1} + dt*k1
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1

        # Term 1: H2 · (dz/dy · w) weighted by dz/dy
        scaled_wvec = dz_dy @ wvec
        h2_scaled = self._hvp_residual.state_state_hvp(z, adj_state, scaled_wvec)
        h2_scaled_flat = self._bkd.flatten(h2_scaled)
        k2_term1 = dz_dy.T @ h2_scaled_flat

        # Term 2: J2 · dt · H1 · w (with adjoint = J2^T · adj)
        J2_T_adj = J2.T @ adj_state
        k2_term2 = dt * self._hvp_residual.state_state_hvp(fsol_nm1, J2_T_adj, wvec)

        result = self._bkd.flatten(k1_ss_hvp) + k2_term1 + self._bkd.flatten(k2_term2)
        return -0.5 * dt * result

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R / dy_{n-1} \, dp) v` contracted with adjoint.

        For :math:`k_2 = f(z, p)` where :math:`z = y + \Delta t \, k_1(y, p)`:

        .. math::

            \frac{d^2 R}{dy \, dp} = -\frac{\Delta t}{2}
            \left(\frac{d^2 k_1}{dy \, dp}
            + \frac{d^2 k_2}{dy \, dp}\right)

        The :math:`k_2` contribution has three terms from the chain rule:

        - Term 1: :math:`(\partial^2 f / \partial p \partial z)|_z \cdot dz/dy`
        - Term 2: :math:`H_z \cdot dz/dy \cdot dz/dp`
        - Term 3: :math:`J_z \cdot \Delta t \cdot (\partial^2 k_1 / \partial p \partial y)`
        """
        dt = self._deltat

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        dk1_dp = self._adjoint_residual.param_jacobian(fsol_nm1)

        k1_sp_hvp = self._hvp_residual.state_param_hvp(fsol_nm1, adj_state, vvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1
        dz_dp_v = dt * (dk1_dp @ vvec)
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: dz/dy^T · state_param_hvp(z, adj, v)
        sp_hvp_z = self._hvp_residual.state_param_hvp(z, adj_state, vvec)
        k2_term1 = dz_dy.T @ self._bkd.flatten(sp_hvp_z)

        # Term 2: dz/dy^T · state_state_hvp(z, adj, dz/dp · v)
        ss_hvp_z = self._hvp_residual.state_state_hvp(z, adj_state, dz_dp_v_flat)
        k2_term2 = dz_dy.T @ self._bkd.flatten(ss_hvp_z)

        # Term 3: (J_z^T · adj)^T · dt · state_param_hvp at y
        J2_T_adj = J2.T @ adj_state
        k2_term3 = dt * self._hvp_residual.state_param_hvp(fsol_nm1, J2_T_adj, vvec)

        result = (
            self._bkd.flatten(k1_sp_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R / dp \, dy_{n-1}) w` contracted with adjoint.

        .. math::

            \frac{d^2 R}{dp \, dy} = -\frac{\Delta t}{2}
            \left(\frac{d^2 k_1}{dp \, dy}
            + \frac{d^2 k_2}{dp \, dy}\right)

        The :math:`k_2` contribution:

        - Term 1: :math:`(\partial^2 f / \partial p \partial z)|_z \cdot dz/dy`
        - Term 2: :math:`H_z \cdot dz/dy \cdot \Delta t \cdot df/dp|_y`
        - Term 3: :math:`J_z \cdot \Delta t \cdot (\partial^2 f / \partial p \partial y)|_y`
        """
        dt = self._deltat
        self._residual.bkd()

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        dk1_dp = self._adjoint_residual.param_jacobian(fsol_nm1)

        k1_ps_hvp = self._hvp_residual.param_state_hvp(fsol_nm1, adj_state, wvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        dz_dy = mass + dt * J1
        dz_dy_w = dz_dy @ wvec

        # Term 1: param_state_hvp(z, adj, dz/dy · w)
        k2_term1 = self._hvp_residual.param_state_hvp(z, adj_state, dz_dy_w)

        # Term 2: H_z · (dz/dy · w) · dt · df/dp|_y
        H_z_dz_dy_w = self._hvp_residual.state_state_hvp(z, adj_state, dz_dy_w)
        k2_term2 = dt * (dk1_dp.T @ self._bkd.reshape(H_z_dz_dy_w, (-1, 1)))

        # Term 3: (J_z^T · adj) · dt · param_state_hvp at y
        J2_T_adj = J2.T @ adj_state
        k2_term3 = dt * self._hvp_residual.param_state_hvp(fsol_nm1, J2_T_adj, wvec)

        result = (
            self._bkd.flatten(k1_ps_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R / dp^2) v` contracted with adjoint.

        .. math::

            \frac{d^2 R}{dp^2} = -\frac{\Delta t}{2}
            \left(\frac{d^2 k_1}{dp^2} + \frac{d^2 k_2}{dp^2}\right)

        The :math:`k_2` contribution has five terms:

        - Term 1: :math:`\partial^2 f / \partial p^2 |_z`
        - Term 2: :math:`(\partial^2 f / \partial p \partial z)|_z \cdot dz/dp`
        - Term 3: :math:`H_z \cdot (dz/dp)^2` (quadratic in :math:`dz/dp`)
        - Term 4: :math:`J_z \cdot \Delta t \cdot \partial^2 f / \partial p^2 |_y`
        - Term 5: :math:`(\partial J_z / \partial p) \cdot \Delta t \cdot dk_1/dp`
        """
        dt = self._deltat

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        dk1_dp = self._adjoint_residual.param_jacobian(fsol_nm1)

        k1_pp_hvp = self._hvp_residual.param_param_hvp(fsol_nm1, adj_state, vvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
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
        k2_term4 = dt * self._hvp_residual.param_param_hvp(fsol_nm1, J2_T_adj, vvec)

        # Term 5: ∂J_z/∂p · v · dt · dk1_dp
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
