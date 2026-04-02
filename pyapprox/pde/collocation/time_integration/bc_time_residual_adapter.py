"""BC-enforcing time residual adapter for collocation methods.

Wraps time stepping residuals and applies collocation boundary conditions
to the residual, Jacobian, and adjoint/sensitivity quantities.

Class Hierarchy
---------------
BCEnforcingForwardResidual
    Wraps SensitivityStepperProtocol: forward solve + sensitivity.
BCEnforcingAdjointResidual
    Wraps AdjointEnabledTimeSteppingResidualProtocol: + adjoint methods.
BCEnforcingHVPResidual
    Wraps HVPEnabledTimeSteppingResidualProtocol: + 4 core HVP methods.
BCEnforcingPrevStepHVPResidual
    Wraps PrevStepHVPEnabledTimeSteppingResidualProtocol: + 3 prev_* HVPs.

Use ``create_bc_enforcing_residual()`` factory to create the appropriate
wrapper based on the inner stepper's protocol level.
"""

from typing import Generic, Tuple, cast, overload

from pyapprox.pde.collocation.physics.base import AbstractPhysics
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.pde.time.protocols.time_stepping import (
    AdjointEnabledTimeSteppingResidualProtocol,
    HVPEnabledTimeSteppingResidualProtocol,
    PrevStepHVPEnabledTimeSteppingResidualProtocol,
    SensitivityStepperProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

# =========================================================================
# Level 1: Forward + Sensitivity
# =========================================================================


class BCEnforcingForwardResidual(Generic[Array]):
    """Wraps a SensitivityStepperProtocol and applies collocation BCs.

    Provides forward solve methods (bkd, set_time, __call__, jacobian,
    linsolve) and sensitivity methods (is_explicit, has_prev_state_hessian,
    sensitivity_off_diag_jacobian, native_residual).

    Parameters
    ----------
    time_residual : SensitivityStepperProtocol
        The underlying time stepping residual.
    physics : AbstractPhysics
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: SensitivityStepperProtocol[Array],
        physics: AbstractPhysics[Array],
        bkd: Backend[Array],
    ) -> None:
        self._inner = time_residual
        self._physics = physics
        self._bkd = bkd
        self._t_np1 = 0.0
        bc_class = self._physics.bc_dof_classification()
        self._essential = bc_class.essential
        self._row_replaced = bc_class.row_replaced

    # -- TimeSteppingResidualProtocol --

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_time(self, time: float, deltat: float, prev_state: Array) -> None:
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

    # -- SensitivityStepperProtocol --

    @property
    def native_residual(self) -> ODEResidualProtocol[Array]:
        """Access the underlying ODE residual."""
        return self._inner.native_residual

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        """Forward sensitivity off-diagonal block with BC rows zeroed.

        B_n[b,:] = 0 at row-replaced DOFs (BCs depend only on current
        state). The inner stepper returns the raw -M without BC enforcement,
        so we must zero those rows explicitly.
        """
        result = self._inner.sensitivity_off_diag_jacobian(fsol_nm1, fsol_n, deltat)
        result = self._bkd.copy(result)
        for idx in self._row_replaced:
            result[idx, :] = 0.0
        return result

    def is_explicit(self) -> bool:
        """Return whether scheme is explicit."""
        return bool(self._inner.is_explicit())

    def has_prev_state_hessian(self) -> bool:
        """Return whether R_{n+1} depends on f(y_n)."""
        return bool(self._inner.has_prev_state_hessian())

    # -- BC zeroing helpers --

    def _zero_bc_rows(self, matrix: Array, zero_bc_rows: bool = True) -> Array:
        """Zero rows at row-replaced DOFs.

        Parameters
        ----------
        matrix : Array
            The parameter Jacobian dR/dp or similar.
        zero_bc_rows : bool, default True
            If True, zero rows at row_replaced DOFs. Valid when BCs
            are independent of the parameter vector p.
            Set to False when computing dR/dp_bc.
        """
        if zero_bc_rows:
            matrix = self._bkd.copy(matrix)
            for idx in self._row_replaced:
                if matrix.ndim == 1:
                    matrix[idx] = 0.0
                else:
                    matrix[idx, :] = 0.0
        return matrix

    def zero_adjoint_rhs(self, dqdu: Array, zero_essential: bool = True) -> Array:
        """Zero dQ/dy at essential (Dirichlet) BC DOFs.

        Parameters
        ----------
        dqdu : Array
            Functional derivative dQ/dy at a single time step.
            Shape: (nstates,).
        zero_essential : bool, default True
            If True, zero dQ/dy at essential BC DOFs, forcing
            lambda[b] = 0. Correct when differentiating w.r.t.
            PDE parameters (essential DOFs are prescribed).
            Set to False when computing gradients w.r.t. BC parameters.

        Returns
        -------
        Array
            Copy with essential BC DOFs zeroed. Shape: (nstates,).
        """
        dqdu = self._bkd.copy(dqdu)
        if zero_essential:
            for idx in self._essential:
                dqdu[idx] = 0.0
        return dqdu

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"inner={type(self._inner).__name__}, "
            f"physics={type(self._physics).__name__}, "
            f"essential={self._essential}, "
            f"row_replaced={self._row_replaced})"
        )


# =========================================================================
# Level 2: + Adjoint
# =========================================================================


class BCEnforcingAdjointResidual(BCEnforcingForwardResidual[Array], Generic[Array]):
    """Extends BCEnforcingForwardResidual with adjoint methods.

    Wraps an AdjointEnabledTimeSteppingResidualProtocol.

    Parameters
    ----------
    time_residual : AdjointEnabledTimeSteppingResidualProtocol
        The underlying time stepping residual with adjoint support.
    physics : AbstractPhysics
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: AdjointEnabledTimeSteppingResidualProtocol[Array],
        physics: AbstractPhysics[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(time_residual, physics, bkd)

    @property
    def _adjoint_inner(self) -> AdjointEnabledTimeSteppingResidualProtocol[Array]:
        """Typed accessor for inner as AdjointEnabled. Safe via constructor."""
        return cast(AdjointEnabledTimeSteppingResidualProtocol[Array], self._inner)

    @property
    def native_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Access the underlying ODE residual (narrowed to ParamJacobian)."""
        return self._adjoint_inner.native_residual

    def param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """Compute parameter Jacobian dR/dp with BC corrections."""
        result = self._adjoint_inner.param_jacobian(fsol_nm1, fsol_n)
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
                phys_sens = self._build_bc_physical_sensitivities(
                    bc, fsol_n, self._t_np1
                )
                result = bc.apply_to_param_jacobian(
                    result,
                    fsol_n,
                    self._t_np1,
                    physical_sensitivities=phys_sens,
                )
        else:
            result = self._zero_bc_rows(result)
        return result

    def _build_bc_physical_sensitivities(
        self, bc: object, state_1d: Array, time: float
    ) -> object:
        """Build physical sensitivities dict for one BC's param_jacobian.

        Delegates d(flux·n)/dp computation to the ODE residual adapter via
        bc_flux_param_sensitivity. Only applies to BCs whose normal operator
        has coefficient dependence (e.g., flux Neumann with parameterized D).
        """
        native = self._adjoint_inner.native_residual
        if not hasattr(native, "bc_flux_param_sensitivity"):
            return None
        if not hasattr(bc, "normal_operator"):
            return None
        normal_op = bc.normal_operator()
        if not (
            hasattr(normal_op, "has_coefficient_dependence")
            and normal_op.has_coefficient_dependence()
        ):
            return None
        bc_idx = bc.boundary_indices()
        normals = normal_op.normals()
        dflux_n_dp = native.bc_flux_param_sensitivity(state_1d, time, bc_idx, normals)
        if dflux_n_dp is None:
            return None
        return {"dflux_n_dp": dflux_n_dp}

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """Adjoint diagonal block: transpose of BC-enforced forward Jacobian.

        self.jacobian() calls physics.apply_boundary_conditions(), producing
        the correct BC-enforced forward Jacobian for ALL BC types and ALL
        solver types. Its transpose is the correct adjoint diagonal block.
        """
        return self.jacobian(fsol_n).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """Adjoint off-diagonal block: B_{n+1}^T with BC columns zeroed.

        B_n[b,:] = 0 at DOFs where the solver replaced the residual row.
        In the transpose, this means B_n^T[:,b] = 0 -- zero columns, not rows.
        """
        result = self._adjoint_inner.adjoint_off_diag_jacobian(fsol_n, deltat_np1)
        result = self._bkd.copy(result)
        for idx in self._row_replaced:
            result[:, idx] = 0.0
        return result

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """Compute adjoint IC using BC-enforced Jacobian.

        The inner adjoint_initial_condition uses the inner Jacobian
        (without BC enforcement), giving wrong lambda at ALL DOFs.
        We must solve with the BC-wrapped adjoint_diag_jacobian instead.
        """
        final_dqdu = self.zero_adjoint_rhs(final_dqdu)
        drduT_diag = self.adjoint_diag_jacobian(final_fwd_sol)
        return self._bkd.solve(drduT_diag, -final_dqdu)

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """Compute adjoint at initial time using BC-enforced mass matrix.

        Delegates mass matrix modification to the physics layer via
        apply_bc_to_mass(), which applies identity rows/columns at
        essential (Dirichlet) DOFs only.
        """
        dqdu_0 = self.zero_adjoint_rhs(dqdu_0)
        mass = self._adjoint_inner.native_residual.mass_matrix(fsol_0.shape[0]).T
        mass = self._physics.apply_bc_to_mass(mass)
        drduT_offdiag = self.adjoint_off_diag_jacobian(fsol_0, deltat_1)
        return self._bkd.solve(mass, -drduT_offdiag @ asol_1 - dqdu_0)

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """Compute quadrature rule consistent with time discretization."""
        return self._adjoint_inner.quadrature_samples_weights(times)

    def initial_param_jacobian(self) -> Array:
        """Compute initial condition param Jacobian with BC rows zeroed."""
        result = self._adjoint_inner.native_residual.initial_param_jacobian()
        return self._zero_bc_rows(result)


# =========================================================================
# Level 3: + Core HVP (4 methods)
# =========================================================================


class BCEnforcingHVPResidual(BCEnforcingAdjointResidual[Array], Generic[Array]):
    """Extends BCEnforcingAdjointResidual with four core HVP methods.

    Wraps an HVPEnabledTimeSteppingResidualProtocol.

    Parameters
    ----------
    time_residual : HVPEnabledTimeSteppingResidualProtocol
        The underlying time stepping residual with HVP support.
    physics : AbstractPhysics
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: HVPEnabledTimeSteppingResidualProtocol[Array],
        physics: AbstractPhysics[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(time_residual, physics, bkd)

    @property
    def _hvp_inner(self) -> HVPEnabledTimeSteppingResidualProtocol[Array]:
        """Typed accessor for inner as HVPEnabled. Safe via constructor."""
        return cast(HVPEnabledTimeSteppingResidualProtocol[Array], self._inner)

    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dy_n^2)w contracted with adjoint, BC entries zeroed.

        Second derivatives of replaced BC rows are zero for all BC types.
        """
        result = self._hvp_inner.state_state_hvp(fsol_nm1, fsol_n, adj_state, wvec)
        result = self._bkd.copy(result)
        for idx in self._row_replaced:
            result[idx] = 0.0
        return result

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dy_n dp)v contracted with adjoint, BC entries zeroed.

        Second derivatives of replaced BC rows are zero for all BC types.
        """
        result = self._hvp_inner.state_param_hvp(fsol_nm1, fsol_n, adj_state, vvec)
        result = self._bkd.copy(result)
        for idx in self._row_replaced:
            result[idx] = 0.0
        return result

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R/dp dy_n)w contracted with adjoint."""
        return self._hvp_inner.param_state_hvp(fsol_nm1, fsol_n, adj_state, wvec)

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R/dp^2)v contracted with adjoint."""
        return self._hvp_inner.param_param_hvp(fsol_nm1, fsol_n, adj_state, vvec)


# =========================================================================
# Level 4: + Prev-step HVP (3 methods, CN only)
# =========================================================================


class BCEnforcingPrevStepHVPResidual(BCEnforcingHVPResidual[Array], Generic[Array]):
    """Extends BCEnforcingHVPResidual with cross-step HVP methods.

    Wraps a PrevStepHVPEnabledTimeSteppingResidualProtocol (Crank-Nicolson).

    Parameters
    ----------
    time_residual : PrevStepHVPEnabledTimeSteppingResidualProtocol
        The underlying time stepping residual with prev_* HVP support.
    physics : AbstractPhysics
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.
    """

    def __init__(
        self,
        time_residual: PrevStepHVPEnabledTimeSteppingResidualProtocol[Array],
        physics: AbstractPhysics[Array],
        bkd: Backend[Array],
    ) -> None:
        super().__init__(time_residual, physics, bkd)

    @property
    def _prev_hvp_inner(
        self,
    ) -> PrevStepHVPEnabledTimeSteppingResidualProtocol[Array]:
        """Typed accessor for inner as PrevStepHVPEnabled. Safe via constructor."""
        return cast(
            PrevStepHVPEnabledTimeSteppingResidualProtocol[Array], self._inner
        )

    def prev_state_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n^2) w contracted with adjoint."""
        return self._prev_hvp_inner.prev_state_state_hvp(fsol_n, adj_state, wvec)

    def prev_state_param_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dy_n dp) v contracted with adjoint."""
        return self._prev_hvp_inner.prev_state_param_hvp(fsol_n, adj_state, vvec)

    def prev_param_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        """Compute (d^2R_{n+1}/dp dy_n) w contracted with adjoint."""
        return self._prev_hvp_inner.prev_param_state_hvp(fsol_n, adj_state, wvec)


# =========================================================================
# Factory
# =========================================================================


@overload
def create_bc_enforcing_residual(
    inner: PrevStepHVPEnabledTimeSteppingResidualProtocol[Array],
    physics: AbstractPhysics[Array],
    bkd: Backend[Array],
) -> BCEnforcingPrevStepHVPResidual[Array]: ...


@overload
def create_bc_enforcing_residual(
    inner: HVPEnabledTimeSteppingResidualProtocol[Array],
    physics: AbstractPhysics[Array],
    bkd: Backend[Array],
) -> BCEnforcingHVPResidual[Array]: ...


@overload
def create_bc_enforcing_residual(
    inner: AdjointEnabledTimeSteppingResidualProtocol[Array],
    physics: AbstractPhysics[Array],
    bkd: Backend[Array],
) -> BCEnforcingAdjointResidual[Array]: ...


@overload
def create_bc_enforcing_residual(
    inner: SensitivityStepperProtocol[Array],
    physics: AbstractPhysics[Array],
    bkd: Backend[Array],
) -> BCEnforcingForwardResidual[Array]: ...


def create_bc_enforcing_residual(
    inner: SensitivityStepperProtocol[Array],
    physics: AbstractPhysics[Array],
    bkd: Backend[Array],
) -> BCEnforcingForwardResidual[Array]:
    """Create a BC-enforcing wrapper at the appropriate protocol level.

    Checks most-specific protocol first, returning the richest wrapper
    that the inner stepper supports.

    Parameters
    ----------
    inner : SensitivityStepperProtocol
        The time stepping residual to wrap.
    physics : AbstractPhysics
        Collocation physics with boundary conditions.
    bkd : Backend
        Computational backend.

    Returns
    -------
    BCEnforcingForwardResidual
        The appropriate BC wrapper (may be a subclass).
    """
    if isinstance(inner, PrevStepHVPEnabledTimeSteppingResidualProtocol):
        return BCEnforcingPrevStepHVPResidual(inner, physics, bkd)
    if isinstance(inner, HVPEnabledTimeSteppingResidualProtocol):
        return BCEnforcingHVPResidual(inner, physics, bkd)
    if isinstance(inner, AdjointEnabledTimeSteppingResidualProtocol):
        return BCEnforcingAdjointResidual(inner, physics, bkd)
    return BCEnforcingForwardResidual(inner, physics, bkd)
