"""BC-enforcing time residual adapter for collocation methods.

Wraps a TimeSteppingResidualBase (BackwardEuler, CrankNicolson, ForwardEuler,
Heun) and applies collocation boundary conditions to the residual, Jacobian,
and adjoint/sensitivity quantities.

This bridges the gap between TimeIntegrator (which expects BCs to be handled
externally) and collocation physics (which enforces BCs by replacing rows in
the Newton residual/Jacobian).
"""

from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.time.protocols.base import TimeSteppingResidualBase
from pyapprox.typing.pde.time.protocols.ode_residual import ODEResidualProtocol


class BCEnforcingTimeResidual(Generic[Array]):
    """Wraps a TimeSteppingResidualBase and applies collocation BCs.

    Uses dynamic method binding: adjoint, param_jacobian, HVP, and
    prev_* methods are only exposed if the wrapped stepper has them.

    Parameters
    ----------
    time_residual : TimeSteppingResidualBase
        The underlying time stepping residual (e.g., BackwardEulerResidual).
    physics : PhysicsProtocol
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: TimeSteppingResidualBase[Array],
        physics,
        bkd: Backend[Array],
    ):
        self._inner = time_residual
        self._physics = physics
        self._bkd = bkd
        self._t_np1 = 0.0
        self._bc_indices = self._collect_bc_indices()
        self._setup_methods()

    def _collect_bc_indices(self):
        """Collect all boundary DOF indices from physics BCs."""
        indices = []
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
                bc_idx = bc.boundary_indices()
                for ii in range(bc_idx.shape[0]):
                    indices.append(int(bc_idx[ii]))
        return indices

    def _setup_methods(self):
        """Dynamic binding based on wrapped stepper capabilities."""
        if hasattr(self._inner, "param_jacobian"):
            self.param_jacobian = self._param_jacobian_impl
            self.adjoint_diag_jacobian = self._adjoint_diag_jacobian_impl
            self.adjoint_off_diag_jacobian = (
                self._adjoint_off_diag_jacobian_impl
            )
            self.adjoint_initial_condition = (
                self._adjoint_initial_condition_impl
            )
            if hasattr(self._inner, "initial_param_jacobian"):
                self.initial_param_jacobian = (
                    self._initial_param_jacobian_impl
                )
        if hasattr(self._inner, "state_state_hvp"):
            self.state_state_hvp = self._state_state_hvp_impl
            self.state_param_hvp = self._state_param_hvp_impl
            self.param_state_hvp = self._param_state_hvp_impl
            self.param_param_hvp = self._param_param_hvp_impl
        if hasattr(self._inner, "prev_state_state_hvp"):
            self.prev_state_state_hvp = self._inner.prev_state_state_hvp
            self.prev_state_param_hvp = self._inner.prev_state_param_hvp
            self.prev_param_state_hvp = self._inner.prev_param_state_hvp

    # =========================================================================
    # TimeSteppingResidualProtocol (always present)
    # =========================================================================

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_time(
        self, time: float, deltat: float, prev_state: Array
    ) -> None:
        """Set time stepping context and track t_{n+1}."""
        self._inner.set_time(time, deltat, prev_state)
        self._t_np1 = time + deltat

    def __call__(self, state: Array) -> Array:
        """Evaluate residual with BCs applied at t_{n+1}."""
        residual = self._inner(state)
        jacobian = self._inner.jacobian(state)
        residual, _ = self._physics.apply_boundary_conditions(
            residual, jacobian, state, self._t_np1
        )
        return residual

    def jacobian(self, state: Array) -> Array:
        """Compute Jacobian with BCs applied at t_{n+1}."""
        residual = self._inner(state)
        jacobian = self._inner.jacobian(state)
        _, jacobian = self._physics.apply_boundary_conditions(
            residual, jacobian, state, self._t_np1
        )
        return jacobian

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve J dy = residual using BC-modified Jacobian."""
        jac = self.jacobian(state)
        return self._bkd.solve(jac, residual)

    # =========================================================================
    # Sensitivity protocol (always present, pure delegation)
    # =========================================================================

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Access the underlying ODE residual."""
        return self._inner.native_residual

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """Compute dR_n/dy_{n-1} for forward sensitivity."""
        return self._inner.sensitivity_off_diag_jacobian(
            fsol_nm1, fsol_n, deltat
        )

    def is_explicit(self) -> bool:
        """Return whether scheme is explicit."""
        return self._inner.is_explicit()

    def has_prev_state_hessian(self) -> bool:
        """Return whether R_{n+1} depends on f(y_n)."""
        return self._inner.has_prev_state_hessian()

    # =========================================================================
    # Template methods (always present, pure delegation)
    # =========================================================================

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """Compute adjoint at initial time (final backward step)."""
        return self._inner.adjoint_final_solution(
            fsol_0, asol_1, dqdu_0, deltat_1
        )

    def quadrature_samples_weights(
        self, times: Array
    ) -> Tuple[Array, Array]:
        """Compute quadrature rule consistent with time discretization."""
        return self._inner.quadrature_samples_weights(times)

    # =========================================================================
    # BC zeroing helper
    # =========================================================================

    def _zero_bc_rows(self, matrix: Array) -> Array:
        """Zero rows of a matrix at boundary DOF indices."""
        matrix = self._bkd.copy(matrix)
        for idx in self._bc_indices:
            if matrix.ndim == 1:
                matrix[idx] = 0.0
            else:
                matrix[idx, :] = 0.0
        return matrix

    # =========================================================================
    # Adjoint methods (conditionally bound)
    # =========================================================================

    def _param_jacobian_impl(
        self, fsol_nm1: Array, fsol_n: Array
    ) -> Array:
        """Compute parameter Jacobian dR/dp with BC rows zeroed."""
        result = self._inner.param_jacobian(fsol_nm1, fsol_n)
        return self._zero_bc_rows(result)

    def _adjoint_diag_jacobian_impl(self, fsol_n: Array) -> Array:
        """Compute (dR/dy_n)^T with BC rows/cols zeroed."""
        result = self._inner.adjoint_diag_jacobian(fsol_n)
        result = self._zero_bc_rows(result)
        result = self._bkd.copy(result)
        for idx in self._bc_indices:
            result[:, idx] = 0.0
            result[idx, idx] = 1.0
        return result

    def _adjoint_off_diag_jacobian_impl(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """Compute off-diagonal adjoint coupling with BC rows zeroed."""
        result = self._inner.adjoint_off_diag_jacobian(fsol_n, deltat_np1)
        return self._zero_bc_rows(result)

    def _adjoint_initial_condition_impl(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """Compute adjoint IC with BC DOFs zeroed."""
        result = self._inner.adjoint_initial_condition(
            final_fwd_sol, final_dqdu
        )
        result = self._bkd.copy(result)
        for idx in self._bc_indices:
            result[idx] = 0.0
        return result

    def _initial_param_jacobian_impl(self) -> Array:
        """Compute initial condition param Jacobian with BC rows zeroed."""
        result = self._inner.initial_param_jacobian()
        return self._zero_bc_rows(result)

    # =========================================================================
    # HVP methods (conditionally bound)
    # =========================================================================

    def _state_state_hvp_impl(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dy_n^2)w contracted with adjoint, BC entries zeroed."""
        result = self._inner.state_state_hvp(
            fsol_nm1, fsol_n, adj_state, wvec
        )
        result = self._bkd.copy(result)
        for idx in self._bc_indices:
            result[idx] = 0.0
        return result

    def _state_param_hvp_impl(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dy_n dp)v contracted with adjoint, BC entries zeroed."""
        result = self._inner.state_param_hvp(
            fsol_nm1, fsol_n, adj_state, vvec
        )
        result = self._bkd.copy(result)
        for idx in self._bc_indices:
            result[idx] = 0.0
        return result

    def _param_state_hvp_impl(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dp dy_n)w contracted with adjoint."""
        return self._inner.param_state_hvp(
            fsol_nm1, fsol_n, adj_state, wvec
        )

    def _param_param_hvp_impl(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dp^2)v contracted with adjoint."""
        return self._inner.param_param_hvp(
            fsol_nm1, fsol_n, adj_state, vvec
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={type(self._inner).__name__}, "
            f"physics={type(self._physics).__name__}, "
            f"bc_indices={self._bc_indices})"
        )
