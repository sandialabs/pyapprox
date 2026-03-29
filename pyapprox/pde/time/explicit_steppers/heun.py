"""
Heun's method (RK2) time stepping residual with adjoint support.

Heun's method is a second-order explicit Runge-Kutta method:

    k1 = f(y_{n-1}, t_{n-1})
    k2 = f(y_{n-1} + Δt·k1, t_n)
    M·(y_n - y_{n-1}) = (Δt/2)·(k1 + k2)

This is also known as the explicit trapezoidal method or improved Euler method.

This module provides full adjoint support for gradient computation dQ/dp
via the adjoint method.
"""

from pyapprox.pde.time.protocols import (
    HVPCapableMixin,
    ParamJacobianCapableMixin,
    TimeSteppingResidualBase,
)
from pyapprox.util.backends.protocols import Array


class HeunResidual(
    ParamJacobianCapableMixin[Array],
    HVPCapableMixin[Array],
    TimeSteppingResidualBase[Array],
):
    """
    Heun's method (RK2) time stepping residual.

    Two-stage explicit Runge-Kutta method (2nd order):

        k1 = f(y_{n-1}, t_{n-1})
        k2 = f(y_{n-1} + Δt·k1, t_n)
        y_n = y_{n-1} + (Δt/2)·(k1 + k2)

    Residual: R(y_n) = M·(y_n - y_{n-1}) - (Δt/2)·(k1 + k2) = 0

    Optional Methods
    ----------------
    The following methods are conditionally available based on the
    underlying ODE residual capabilities:

    - ``param_jacobian(fsol_nm1, fsol_n)``: Available if ODE has ``param_jacobian``
    - ``state_state_hvp(...)``, etc.: Available if ODE has HVP methods

    Check availability with ``hasattr(residual, 'param_jacobian')``.
    """

    def __call__(self, state: Array) -> Array:
        """
        Evaluate the Heun residual.

        R(y_n) = M·(y_n - y_{n-1}) - (Δt/2)·(k1 + k2)

        where:
            k1 = f(y_{n-1}, t_{n-1})
            k2 = f(y_{n-1} + Δt·k1, t_n)

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Residual value. Shape: (nstates,)
        """
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
        """
        Compute the Jacobian dR/dy_n.

        For Heun's method (explicit), dR/dy_n = M (mass matrix) since the
        residual is linear in y_n.

        Parameters
        ----------
        state : Array
            State at current time step y_n. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian = M. Shape: (nstates, nstates)
        """
        return self._residual.mass_matrix(state.shape[0])

    # =========================================================================
    # Sensitivity Protocol Methods
    # =========================================================================

    def is_explicit(self) -> bool:
        """Return True since Heun is an explicit scheme."""
        return True

    def has_prev_state_hessian(self) -> bool:
        """Return False since R_{n+1} does not depend on f(y_n)."""
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """
        Compute dR_n/dy_{n-1} for forward sensitivity propagation.

        For Heun R_n = y_n - y_{n-1} - (Δt/2)·(k1 + k2):
            k1 = f(y_{n-1}), k2 = f(y_{n-1} + Δt·k1)

        dR_n/dy_{n-1} = -(M + (Δt/2)·(J1 + J2·(M + Δt·J1)))

        Parameters
        ----------
        fsol_nm1 : Array
            Solution at previous time step y_{n-1}. Shape: (nstates,)
        fsol_n : Array
            Solution at current time step y_n. Shape: (nstates,)
        deltat : float
            Time step size Δt.

        Returns
        -------
        Array
            Off-diagonal Jacobian dR_n/dy_{n-1}. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time)
        k1_jac = self._residual.jacobian(fsol_nm1)

        # k2 evaluation point
        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + deltat * k1

        self._residual.set_time(self._time + deltat)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix(fsol_nm1.shape[0])

        # dR/dy_{n-1} = -(M + (Δt/2)·(J1 + J2·(M + Δt·J1)))
        return -(mass + 0.5 * deltat * (k1_jac + k2_jac @ (mass + deltat * k1_jac)))

    # =========================================================================
    # Adjoint Methods
    # =========================================================================

    def _param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """
        Compute the parameter Jacobian dR/dp for one time step.

        For Heun's method with k2 = f(y_{n-1} + Δt·k1, t_n):

        dR/dp = -(Δt/2)·[dk1/dp + dk2/dp]

        where:
            dk1/dp = ∂f/∂p|_{y_{n-1}, t_{n-1}}
            dk2/dp = ∂f/∂p|_{k2_state, t_n} + ∂f/∂y|_{k2_state, t_n} · Δt · dk1/dp

        Parameters
        ----------
        fsol_nm1 : Array
            Forward solution at previous time step. Shape: (nstates,)
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        # k1 stage
        self._residual.set_time(self._time)
        k1_param_jac = self._param_residual().param_jacobian(fsol_nm1)

        # k2 stage: k2_state = y_{n-1} + Δt·k1
        k1 = self._residual(fsol_nm1)
        k2_state = fsol_nm1 + self._deltat * k1

        self._residual.set_time(self._time + self._deltat)
        k2_state_jac = self._residual.jacobian(k2_state)
        k2_param_jac = self._param_residual().param_jacobian(k2_state)

        # Chain rule: dk2/dp = ∂f/∂p + ∂f/∂y · Δt · dk1/dp
        jac = -(
            0.5
            * self._deltat
            * (
                k1_param_jac
                + k2_param_jac
                + self._deltat * (k2_state_jac @ k1_param_jac)
            )
        )
        return jac

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """
        Compute the diagonal Jacobian block for adjoint solve.

        For Heun (explicit), this is just Mᵀ.

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)

        Returns
        -------
        Array
            Mᵀ. Shape: (nstates, nstates)
        """
        return self._residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_off_diag_jacobian(self, fsol_n: Array, deltat_np1: float) -> Array:
        """
        Compute the off-diagonal Jacobian for adjoint coupling.

        For Heun's method:
        dR_{n+1}/dy_n involves derivatives through both k1 and k2 stages.

        d/dy_n [-(Δt/2)·(k1 + k2)]
        where k1 = f(y_n), k2 = f(y_n + Δt·k1)

        = -(Δt/2)·[J_1 + J_2·(I + Δt·J_1)]

        Parameters
        ----------
        fsol_n : Array
            Forward solution at current time step. Shape: (nstates,)
        deltat_np1 : float
            Time step size for the next interval.

        Returns
        -------
        Array
            Off-diagonal coupling. Shape: (nstates, nstates)
        """
        self._residual.set_time(self._time)
        k1_jac = self._residual.jacobian(fsol_n)

        # k2 evaluation point
        k1 = self._residual(fsol_n)
        k2_state = fsol_n + deltat_np1 * k1

        self._residual.set_time(self._time + deltat_np1)
        k2_jac = self._residual.jacobian(k2_state)

        mass = self._residual.mass_matrix(fsol_n.shape[0])

        # dR/dy_{n-1} for Heun: -(M + (Δt/2)·(J_1 + J_2·(I + Δt·J_1)))
        jac = -(
            mass + 0.5 * deltat_np1 * (k1_jac + k2_jac @ (mass + deltat_np1 * k1_jac))
        )
        return jac.T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """
        Compute initial condition for backward adjoint solve.

        For explicit methods, λ_N = -dQ/dy_N (no linear solve needed).

        Parameters
        ----------
        final_fwd_sol : Array
            Forward solution at final time. Shape: (nstates,)
        final_dqdu : Array
            Gradient dQ/dy at final time. Shape: (nstates,)

        Returns
        -------
        Array
            Adjoint solution at final time λ_N. Shape: (nstates,)
        """
        return -final_dqdu

    def _get_quadrature_class(self) -> type:
        """Return quadrature class for Heun's method (trapezoidal/linear)."""
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseLinear,
        )

        return PiecewiseLinear

    # =========================================================================
    # HVP Methods (conditionally available via dynamic binding)
    # =========================================================================

    def _state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dy_{n-1}²)·w contracted with adjoint.

        For Heun: involves second derivatives through both k1 and k2 stages
        with proper chain rule.

        R = y_n - y_{n-1} - (dt/2)(k1 + k2)
        k1 = f(y_{n-1}), k2 = f(z) where z = y_{n-1} + dt*k1

        dR/dy = -I - (dt/2)(J1 + J2*(I + dt*J1))

        d²R/dy² = -(dt/2)(H1 + H2*(I+dt*J1)² + J2*dt*H1)

        where H1, H2 = d²f/dy² at y_{n-1} and z respectively.
        """
        dt = self._deltat

        # Stage 1: k1 = f(y_{n-1})
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])

        # d²k1/dy² · w = H1 · w
        k1_ss_hvp = self._hvp_residual().state_state_hvp(fsol_nm1, adj_state, wvec)

        # Stage 2: k2 = f(z) where z = y_{n-1} + dt*k1
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        # dz/dy = I + dt*J1
        dz_dy = mass + dt * J1

        # d²k2/dy² · w involves chain rule:
        # d(J2 * dz/dy)/dy · w = H2 · (dz/dy)² · w + J2 · dt · H1 · w
        #
        # Term 1: H2 · (dz/dy · w) weighted by (dz/dy)
        # We need to compute: adj^T · H2 · (dz/dy)² · w
        # = (adj^T · H2 · dz/dy) · (dz/dy · w)
        #
        # For scalar case: dz_dy * w gives the effective w for H2
        scaled_wvec = dz_dy @ wvec
        # H2 · (dz/dy · w)
        h2_scaled = self._hvp_residual().state_state_hvp(z, adj_state, scaled_wvec)
        # Flatten for consistent shapes
        h2_scaled_flat = self._bkd.flatten(h2_scaled)
        # Then multiply by dz/dy^T (for the outer chain rule)
        k2_term1 = dz_dy.T @ h2_scaled_flat

        # Term 2: J2 · dt · H1 · w
        # = J2 · dt · k1_ss_hvp (where k1_ss_hvp = H1 · w already contracted with adj)
        # But we need: adj^T · J2 · dt · H1 · w
        # Note: k1_ss_hvp is already adj^T · H1 · w, so this is different
        # We need: (J2^T · adj)^T · (dt · H1 · w) = (J2^T · adj)^T · dt · H1 · w
        # So we need to compute H1 · w with adjoint = J2^T · adj
        J2_T_adj = J2.T @ adj_state
        k2_term2 = dt * self._hvp_residual().state_state_hvp(fsol_nm1, J2_T_adj, wvec)

        # Flatten k1_ss_hvp for consistent addition
        result = self._bkd.flatten(k1_ss_hvp) + k2_term1 + self._bkd.flatten(k2_term2)
        return -0.5 * dt * result

    def _state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dy_{n-1} dp)·v contracted with adjoint.

        R = y_n - y_{n-1} - (dt/2)(k1 + k2)
        k1 = f(y, p), k2 = f(z, p) where z = y + dt*k1(y,p)

        d²R/(dy dp) = -(dt/2)(d²k1/(dy dp) + d²k2/(dy dp))

        For k2 = f(z, p) where z = y + dt·k1(y,p):
            dk2/dp = ∂f/∂p|_z + J_z · dz/dp

            d(dk2/dp)/dy involves:
            - Term 1: (∂²f/∂p∂z)|_z · dz/dy     (chain rule on ∂f/∂p|_z)
            - Term 2: H_z · dz/dy · dz/dp        (chain rule on J_z through z)
            - Term 3: J_z · dt · (∂²k1/∂p∂y)    (chain rule on dz/dp)

        The contracted HVP: adj^T · d²k2/(dy dp) · v involves applying dz/dy
        AFTER contracting with the Hessian/mixed derivatives, not before.
        """
        dt = self._deltat

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        dk1_dp = self._param_residual().param_jacobian(fsol_nm1)

        # d²k1/(dy dp) · v
        k1_sp_hvp = self._hvp_residual().state_param_hvp(fsol_nm1, adj_state, vvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        # dz/dy = I + dt*J1
        dz_dy = mass + dt * J1
        # dz/dp = dt * dk1/dp
        dz_dp_v = dt * (dk1_dp @ vvec)
        # Flatten for state_state_hvp which expects 1D wvec
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # d²k2/(dy dp) · v involves:
        # Term 1: adj^T · (∂²f/∂p∂z)|_z · dz/dy · v
        # = dz/dy^T · state_param_hvp(z, adj, v)
        sp_hvp_z = self._hvp_residual().state_param_hvp(z, adj_state, vvec)
        k2_term1 = dz_dy.T @ self._bkd.flatten(sp_hvp_z)

        # Term 2: adj^T · H_z · dz/dy · (dz/dp · v)
        # = dz/dy^T · state_state_hvp(z, adj, dz/dp · v)
        ss_hvp_z = self._hvp_residual().state_state_hvp(z, adj_state, dz_dp_v_flat)
        k2_term2 = dz_dy.T @ self._bkd.flatten(ss_hvp_z)

        # Term 3: J_z · dt · (∂²k1/∂p∂y) · v
        # adj^T · J_z · dt · (∂²k1/∂p∂y) · v
        # = (J_z^T · adj)^T · dt · (∂²k1/∂p∂y) · v
        J2_T_adj = J2.T @ adj_state
        k2_term3 = dt * self._hvp_residual().state_param_hvp(fsol_nm1, J2_T_adj, vvec)

        # Flatten all terms for consistent shape
        result = (
            self._bkd.flatten(k1_sp_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result

    def _param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp dy_{n-1})·w contracted with adjoint.

        R = y_n - y_{n-1} - (dt/2)(k1 + k2)
        k1 = f(y, p), k2 = f(z, p) where z = y + dt*k1(y,p)

        dR/dp = -(dt/2)(dk1/dp + dk2/dp)

        d²R/(dp dy) = -(dt/2)(d²k1/(dp dy) + d²k2/(dp dy))

        For k2 = f(z, p) where z = y + dt*k1(y,p):
            dk2/dp = ∂f/∂p|_z + J_z · dz/dp
                   = ∂f/∂p|_z + J_z · dt · df/dp|_y

            d/dy[dk2/dp] = (∂²f/∂p∂z)|_z · dz/dy           -- Term 1
                         + H_z · dz/dy · dt · df/dp|_y     -- Term 2 (chain rule)
                         + J_z · dt · (∂²f/∂p∂y)|_y        -- Term 3
        """
        dt = self._deltat
        self._residual.bkd()

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        J1 = self._residual.jacobian(fsol_nm1)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        dk1_dp = self._param_residual().param_jacobian(fsol_nm1)

        # d²k1/(dp dy) · w
        k1_ps_hvp = self._hvp_residual().param_state_hvp(fsol_nm1, adj_state, wvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        # dz/dy = I + dt*J1
        dz_dy = mass + dt * J1
        # dz/dy · w
        dz_dy_w = dz_dy @ wvec

        # Term 1: (∂²f/∂p∂z)|_z · (dz/dy · w)
        # contracted with adj: adj^T · term1
        k2_term1 = self._hvp_residual().param_state_hvp(z, adj_state, dz_dy_w)

        # Term 2: H_z · (dz/dy · w) · dt · df/dp|_y
        # contracted with adj: adj^T · H_z · (dz/dy · w) gives scalar/vector
        # then multiply by dt · df/dp|_y^T to get (nparams,) result
        # = dt · (state_state_hvp(z, adj, dz_dy_w)) · df/dp|_y^T
        H_z_dz_dy_w = self._hvp_residual().state_state_hvp(z, adj_state, dz_dy_w)
        # H_z_dz_dy_w is scalar (nstates=1) or vector (nstates,)
        # df/dp|_y is (nstates, nparams)
        # Result: df/dp|_y^T · H_z_dz_dy_w = (nparams, nstates) @ (nstates,) =
        # (nparams,)
        k2_term2 = dt * (dk1_dp.T @ self._bkd.reshape(H_z_dz_dy_w, (-1, 1)))

        # Term 3: J_z · dt · (∂²f/∂p∂y)|_y · w
        # contracted with adj: (J_z^T · adj)^T · dt · param_state_hvp at y
        J2_T_adj = J2.T @ adj_state
        k2_term3 = dt * self._hvp_residual().param_state_hvp(fsol_nm1, J2_T_adj, wvec)

        # Flatten all terms for consistent shape
        result = (
            self._bkd.flatten(k1_ps_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
        )
        return -0.5 * dt * result

    def _param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """
        Compute (d²R/dp²)·v contracted with adjoint.

        R = y_n - y_{n-1} - (dt/2)(k1 + k2)
        k1 = f(y, p), k2 = f(z, p) where z = y + dt*k1(y,p)

        dR/dp = -(dt/2)(dk1/dp + dk2/dp)
              = -(dt/2)(df/dp|_y + df/dp|_z + J2 * dt * df/dp|_y)

        d²R/dp² = -(dt/2)(d²k1/dp² + d²k2/dp²)

        For k2 = f(z, p) where z = y + dt*k1:
            dk2/dp = ∂f/∂p|_z + J_z · dt · ∂f/∂p|_y

            d²k2/dp² = d/dp[∂f/∂p|_z] + d/dp[J_z · dt · ∂f/∂p|_y]

            Part A: d/dp[∂f/∂p|_z]
              = ∂²f/∂p²|_z + (∂²f/∂p∂z|_z) · dz/dp       -- Terms 1, 2

            Part B: d/dp[J_z · dt · ∂f/∂p|_y]
              = (dJ_z/dp) · dt · ∂f/∂p|_y                 -- Term 5 (NEW)
              + J_z · dt · ∂²f/∂p²|_y                     -- Term 4

            Where dJ_z/dp = H_z · dz/dp gives:
              = (H_z · dz/dp) · dt · ∂f/∂p|_y             -- Term 5

            And dz/dp = dt · ∂f/∂p|_y, so the quadratic term is:
              = H_z · (dz/dp)²                            -- Term 3
        """
        dt = self._deltat

        # Stage 1
        self._residual.set_time(self._time)
        k1 = self._residual(fsol_nm1)
        dk1_dp = self._param_residual().param_jacobian(fsol_nm1)

        # d²k1/dp² · v
        k1_pp_hvp = self._hvp_residual().param_param_hvp(fsol_nm1, adj_state, vvec)

        # Stage 2
        z = fsol_nm1 + dt * k1
        self._residual.set_time(self._time + dt)
        J2 = self._residual.jacobian(z)

        # dz/dp = dt * dk1/dp  (matrix, shape: nstates x nparams)
        # dz/dp · v = dt * dk1/dp · v  (vector, shape: nstates,)
        dz_dp_v = dt * (dk1_dp @ vvec)
        # Flatten for functions that expect 1D wvec
        dz_dp_v_flat = self._bkd.flatten(dz_dp_v)

        # Term 1: ∂²f/∂p²|_z · v
        k2_term1 = self._hvp_residual().param_param_hvp(z, adj_state, vvec)

        # Term 2: (∂²f/∂p∂z|_z) · dz/dp · v (NO factor of 2!)
        # = param_state_hvp(z, adj, dz_dp_v)
        # NOTE: Previous version had "2 * param_state_hvp" which was incorrect.
        # The factor of 2 was removed because Part A only contributes once.
        k2_term2 = self._hvp_residual().param_state_hvp(z, adj_state, dz_dp_v_flat)

        # Term 3: H_z · (dz/dp)² contribution
        # = adj^T · H_z · (dz/dp · v), weighted by (dz/dp)^T
        # = (dk1_dp)^T · dt · state_state_hvp(z, adj, dz_dp_v)
        h_dz_dp_v = self._hvp_residual().state_state_hvp(z, adj_state, dz_dp_v_flat)
        h_dz_dp_v_flat = self._bkd.flatten(h_dz_dp_v)
        k2_term3 = dt * (dk1_dp.T @ h_dz_dp_v_flat)

        # Term 4: J_z · dt · ∂²f/∂p²|_y · v
        J2_T_adj = J2.T @ adj_state
        k2_term4 = dt * self._hvp_residual().param_param_hvp(fsol_nm1, J2_T_adj, vvec)

        # Term 5 (NEW): ∂J_z/∂p · v · dt · dk1_dp
        # This is the contribution from d/dp[J_z] when differentiating J_z · dt ·
        # ∂f/∂p|_y.
        # dJ_z/dp = ∂J_z/∂p + H_z · dz/dp
        # The ∂J_z/∂p part gives this term (H_z · dz/dp is already in Term 3).
        # ∂J_z/∂p = ∂²f/(∂y∂p)|_z, which is state_param_hvp(z, adj, v).
        # For the HVP, we need: adj · (∂J_z/∂p · v) · dt · dk1_dp
        # = dt · dk1_dp^T @ state_param_hvp(z, adj, v)
        sp_hvp = self._hvp_residual().state_param_hvp(z, adj_state, vvec)
        k2_term5 = dt * (dk1_dp.T @ self._bkd.reshape(sp_hvp, (-1, 1)))

        # Flatten all terms for consistent shape
        result = (
            self._bkd.flatten(k1_pp_hvp)
            + self._bkd.flatten(k2_term1)
            + self._bkd.flatten(k2_term2)
            + self._bkd.flatten(k2_term3)
            + self._bkd.flatten(k2_term4)
            + self._bkd.flatten(k2_term5)
        )
        return -0.5 * dt * result
