"""Parameterized adapter tiers for Galerkin time integration.

Capability is decided ONCE at construction by
:func:`create_galerkin_physics_ode_residual`, which None-checks the
parameterization's :class:`ParamDerivatives` bundle and the system's
:class:`StateDerivatives` bundle to select a fixed-tier adapter class
(never ``hasattr`` or ``isinstance`` on the physics):

- :class:`GalerkinPhysicsToODEResidualAdapter` (base tier, lives in
  ``pyapprox.pde.galerkin.time_integration.physics_adapter``) — raw
  f/J/M, no parameters.
- :class:`GalerkinPhysicsToODEResidualWithSetParamAdapter` — adds
  ``nparams``/``set_param`` (parameterization without derivative
  capability).
- :class:`GalerkinPhysicsToODEResidualWithParamJacobianAdapter` — adds
  ``param_jacobian``/``initial_param_jacobian`` (first-order bundle).
- :class:`GalerkinPhysicsToODEResidualWithHVPAdapter` — adds the three
  parameterization HVP contractions (second-order bundle) and
  ``state_state_hvp`` (from the system's state-derivatives bundle).

Each tier's ``derivatives()`` returns the residual bundle matching its
capability: the stack Jacobian alone for the first two, the parameter
and initial-state Jacobians added for the third, the second derivatives
for the fourth. Parameters live in a space of the discretization's stack.
"""

from typing import Optional, TypeVar, overload

from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    InitialStateFn,
    ODEResidualDerivatives,
    ParamFn,
    dense_operator,
    matrix_free_initial_state_curvature,
    matrix_free_second_derivative,
)
from pyapprox.ode.state_derivatives import StateStateHVPFn
from pyapprox.pde.galerkin.time_integration.neutralized_rows import (
    NeutralizedRows,
    RowTreatmentFactory,
)
from pyapprox.pde.galerkin.time_integration.physics_adapter import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.pde.models.galerkin.constrained_derivatives import (
    constrain_param_derivatives,
    constrain_state_state_hvp,
)
from pyapprox.pde.parameterizations.binding import require_owned_targets
from pyapprox.pde.parameterizations.derivatives import (
    InitialParamHVPFn,
    InitialParamJacobianFn,
    ParamHVPFn,
    ParamJacobianFn,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    LinearSolverProtocol,
    VectorSpaceProtocol,
)

F = TypeVar("F", bound=FormatProtocol)


class GalerkinPhysicsToODEResidualWithSetParamAdapter(
    GalerkinPhysicsToODEResidualAdapter[Array, F]
):
    """Adapter with a parameterization (evaluation-only tier).

    Adds ``nparams``/``set_param`` on top of the base tier. Selected by
    the factory when the parameterization's bundle declares no
    derivative capability.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        The system to adapt, on a stack.
    solver : LinearSolverProtocol
        Solves the implicit system's shifted Jacobians.
    parameterization : ParameterizationProtocol
        Maps parameter vectors to coefficients of objects the system
        holds.
    row_treatment : RowTreatmentFactory, optional
        The essential-row strategy; defaults to ``NeutralizedRows``. The
        parameter derivatives are the same under either strategy: the
        terms they move are independent of the parameters.
    """

    def __init__(
        self,
        discretization: GalerkinStackDiscretization[Array, Array, F],
        solver: LinearSolverProtocol[Array, F],
        parameterization: ParameterizationProtocol[Array],
        row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
    ) -> None:
        self._parameters_version = 0
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        require_owned_targets(parameterization, discretization.system())
        super().__init__(discretization, solver, row_treatment)
        self._parameterization = parameterization
        self._current_params_1d: Optional[Array] = None

    def parameterization(self) -> ParameterizationProtocol[Array]:
        """Return the parameterization."""
        return self._parameterization

    def nparams(self) -> int:
        """Return the number of parameters."""
        return self._parameterization.nparams()

    def set_param(self, param: Array) -> None:
        """Set parameter values through the parameterization.

        Parameters
        ----------
        param : Array
            Parameter vector. Shape: (nparams,) — the ODEResidual
            protocol convention (strictly validated; callers convert at
            the optimizer/ode boundary).
        """
        if param.ndim != 1:
            raise ValueError(
                f"param must be 1D with shape (nparams,), got shape "
                f"{tuple(param.shape)}"
            )
        self._current_params_1d = param
        self._parameterization.apply(param)
        self._parameters_version += 1

    def parameters_version(self) -> int:
        """The number of ``set_param`` calls so far."""
        return self._parameters_version

    def _require_params(self) -> Array:
        """Return the current parameters or raise if set_param not called."""
        if self._current_params_1d is None:
            raise RuntimeError(
                "set_param() must be called before parameter derivatives"
            )
        return self._current_params_1d


class GalerkinPhysicsToODEResidualWithParamJacobianAdapter(
    GalerkinPhysicsToODEResidualWithSetParamAdapter[Array, F]
):
    """Adapter with first-order parameter derivatives.

    Adds ``param_jacobian``/``initial_param_jacobian`` on top of the
    evaluation tier. Selected by the factory when the bundle has both
    parameter jacobians. The bundle is narrowed ONCE here into
    always-present private attributes.
    """

    def __init__(
        self,
        discretization: GalerkinStackDiscretization[Array, Array, F],
        solver: LinearSolverProtocol[Array, F],
        parameterization: ParameterizationProtocol[Array],
        row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
    ) -> None:
        super().__init__(discretization, solver, parameterization, row_treatment)
        derivs = constrain_param_derivatives(
            parameterization.param_derivatives(), self._constraint_set
        )
        param_jacobian = derivs.param_jacobian
        initial_param_jacobian = derivs.initial_param_jacobian
        if param_jacobian is None or initial_param_jacobian is None:
            raise TypeError(
                f"{type(self).__name__} requires a parameterization whose "
                "bundle has param_jacobian and initial_param_jacobian; use "
                "create_galerkin_physics_ode_residual to select the right "
                "tier"
            )
        self._param_jacobian_fn: ParamJacobianFn[Array] = param_jacobian
        self._initial_param_jacobian_fn: InitialParamJacobianFn[Array] = (
            initial_param_jacobian
        )

    def param_jacobian(self, state: Array, time: float) -> Array:
        """Compute the parameter Jacobian of the BC-NEUTRALIZED f.

        The adapter's f carries analytic boundary velocities on
        essential rows (parameter-independent), so those rows of the
        bundle's RAW dF/dp are zeroed (``constrain_param_derivatives``)
        — not left to the BC-enforcing wrapper. Stage-based steppers
        (Heun) compose this Jacobian through M^{-1} and stage Jacobians
        BEFORE the wrapper's final row replacement, so raw essential
        rows would contaminate interior rows of dR/dp.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)
        time : float
            Time t.

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams)
        """
        return self._param_jacobian_fn(state, time, self._require_params())

    def initial_param_jacobian(self) -> Array:
        """Compute d(initial_state)/d(params).

        Returns
        -------
        Array
            Initial-condition Jacobian. Shape: (nstates, nparams)
        """
        return self._initial_param_jacobian_fn(self._require_params())

    def _first_order_parts(
        self,
    ) -> tuple[
        VectorSpaceProtocol[Array],
        VectorSpaceProtocol[Array],
        ParamFn[Array],
        InitialStateFn[Array],
    ]:
        """The state and parameter spaces, both of the discretization's
        stack, and the parameter and initial-state Jacobians as dense
        operators over their matrix methods."""
        states = self._discretization.space()
        params = self._discretization.parameter_space(self.nparams())
        return (
            states,
            params,
            dense_operator(self.param_jacobian, params, states, self._bkd),
            dense_operator(self.initial_param_jacobian, params, states, self._bkd),
        )

    def _build_derivatives(self) -> ODEResidualDerivatives[Array, F]:
        """First order: the stack Jacobian, and the parameter and
        initial-state Jacobians."""
        states, params, param, initial = self._first_order_parts()
        return ODEResidualDerivatives.first_order(
            states,
            params,
            jacobian=self._essential_jacobian.jacobian_operator,
            param=param,
            initial_state=initial,
            affine=self._affine(),
        )


class GalerkinPhysicsToODEResidualWithHVPAdapter(
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter[Array, F]
):
    """Adapter with second-order parameter derivatives.

    Adds the three parameterization HVP contractions (from the
    parameterization's bundle) and ``state_state_hvp`` (from the
    system's state-derivatives bundle) on top of the first-order tier.
    Selected by the factory when both bundles supply them.

    Every contraction represents second derivatives of the
    BC-NEUTRALIZED f (the same function ``__call__``/``jacobian``
    expose): essential rows carry parameter-independent boundary
    velocities, so their second derivatives vanish — the incoming
    adjoint/weight vector is zeroed at essential entries before the
    RAW contraction (``constrain_param_derivatives`` and
    ``constrain_state_state_hvp``). Stage-based steppers (Heun)
    build their own weight vectors (e.g. M^{-T} J2^T lambda, nonzero
    at essential entries) inside the stepper, where the BC-enforcing
    wrapper's adjoint zeroing cannot reach.
    """

    def __init__(
        self,
        discretization: GalerkinStackDiscretization[Array, Array, F],
        solver: LinearSolverProtocol[Array, F],
        parameterization: ParameterizationProtocol[Array],
        row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
    ) -> None:
        state_state_hvp = (
            discretization.system()
            .spatial_operator()
            .state_derivatives()
            .state_state_hvp
        )
        if state_state_hvp is None:
            raise TypeError(
                f"{type(self).__name__} requires a system whose spatial "
                "operator supplies state_state_hvp; use "
                "create_galerkin_physics_ode_residual to select the right "
                "tier"
            )
        super().__init__(discretization, solver, parameterization, row_treatment)
        derivs = constrain_param_derivatives(
            parameterization.param_derivatives(), self._constraint_set
        )
        param_param_hvp = derivs.param_param_hvp
        state_param_hvp = derivs.state_param_hvp
        param_state_hvp = derivs.param_state_hvp
        initial_param_hvp = derivs.initial_param_hvp
        if (
            param_param_hvp is None
            or state_param_hvp is None
            or param_state_hvp is None
            or initial_param_hvp is None
        ):
            raise TypeError(
                f"{type(self).__name__} requires a parameterization "
                "whose bundle has all three HVP contractions and "
                "initial_param_hvp; use create_galerkin_physics_ode_residual "
                "to select the right tier"
            )
        self._state_state_hvp_fn: StateStateHVPFn[Array] = constrain_state_state_hvp(
            state_state_hvp, self._constraint_set
        )
        self._param_param_hvp_fn: ParamHVPFn[Array] = param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = param_state_hvp
        self._initial_param_hvp_fn: InitialParamHVPFn[Array] = initial_param_hvp

    def _build_derivatives(self) -> ODEResidualDerivatives[Array, F]:
        """Second order: the first-order fields, and the second
        derivatives through the vector methods, applied a column at a
        time."""
        states, params, param, initial = self._first_order_parts()
        bkd = self._bkd
        affine = self._affine()
        return ODEResidualDerivatives.second_order(
            states,
            params,
            jacobian=self._essential_jacobian.jacobian_operator,
            param=param,
            initial_state=initial,
            initial_state_curvature=matrix_free_initial_state_curvature(
                self.initial_param_hvp, params, bkd
            ),
            # An affine f has zero state curvature, stated as such.
            state_state=(
                STATED_ZERO
                if affine
                else matrix_free_second_derivative(
                    self.state_state_hvp,
                    self.state_state_hvp,
                    states,
                    states,
                    bkd,
                    True,
                )
            ),
            state_param=matrix_free_second_derivative(
                self.state_param_hvp, self.param_state_hvp, params, states, bkd
            ),
            param_param=matrix_free_second_derivative(
                self.param_param_hvp, self.param_param_hvp, params, params, bkd, True
            ),
            affine=affine,
        )

    def initial_param_hvp(self, weight: Array, vvec: Array) -> Array:
        """sum_i weight_i (d^2 y_0,i/dp^2) v, essential entries zeroed.
        Shape: (nparams,)."""
        return self._initial_param_hvp_fn(self._require_params(), weight, vvec)

    def state_state_hvp(
        self, state: Array, time: float, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dy^2) w. Shape: (nstates,)."""
        return self._state_state_hvp_fn(state, adj_state, wvec, time)

    def param_param_hvp(
        self, state: Array, time: float, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dp^2) v. Shape: (nparams,)."""
        return self._param_param_hvp_fn(
            state, time, self._require_params(), adj_state, vvec
        )

    def state_param_hvp(
        self, state: Array, time: float, adj_state: Array, vvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dy dp) v. Shape: (nstates,)."""
        return self._state_param_hvp_fn(
            state, time, self._require_params(), adj_state, vvec
        )

    def param_state_hvp(
        self, state: Array, time: float, adj_state: Array, wvec: Array
    ) -> Array:
        """Compute lambda^T (d^2F/dp dy) w. Shape: (nparams,)."""
        return self._param_state_hvp_fn(
            state, time, self._require_params(), adj_state, wvec
        )


@overload
def create_galerkin_physics_ode_residual(
    discretization: GalerkinStackDiscretization[Array, Array, F],
    solver: LinearSolverProtocol[Array, F],
    parameterization: None = None,
    row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
) -> GalerkinPhysicsToODEResidualAdapter[Array, F]: ...


@overload
def create_galerkin_physics_ode_residual(
    discretization: GalerkinStackDiscretization[Array, Array, F],
    solver: LinearSolverProtocol[Array, F],
    parameterization: ParameterizationProtocol[Array],
    row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
) -> GalerkinPhysicsToODEResidualWithSetParamAdapter[Array, F]: ...


def create_galerkin_physics_ode_residual(
    discretization: GalerkinStackDiscretization[Array, Array, F],
    solver: LinearSolverProtocol[Array, F],
    parameterization: Optional[ParameterizationProtocol[Array]] = None,
    row_treatment: RowTreatmentFactory[Array] = NeutralizedRows,
) -> GalerkinPhysicsToODEResidualAdapter[Array, F]:
    """Create the widest adapter tier the inputs support.

    Capability enters the stepper stack exactly here: the factory
    None-checks the parameterization's ParamDerivatives bundle and the
    system's StateDerivatives bundle once, then everything above sees
    unconditional fixed-tier methods.

    Essential boundary data are not parameterized: a parameterization
    writes coefficients of the physics, never a boundary condition's
    values, and the derivatives treat those values as independent of
    the parameters. A boundary-value supplier that reads state the
    parameterization writes (by a closure, or by being the same
    coefficient object the physics holds) therefore gives wrong
    derivatives, and neither route can be detected here.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        The system to adapt, on a stack (for skfem systems,
        ``skfem_discretization(system)``).
    solver : LinearSolverProtocol
        Solves the implicit system's shifted Jacobians.
    parameterization : ParameterizationProtocol, optional
        Maps parameter vectors to coefficients of objects the system
        holds.
    row_treatment : RowTreatmentFactory, optional
        The essential-row strategy, passed to whichever tier is chosen;
        defaults to ``NeutralizedRows``.

    Returns
    -------
    GalerkinPhysicsToODEResidualAdapter
        The widest tier supported by the bundles.
    """
    system = discretization.system()
    if parameterization is None:
        return GalerkinPhysicsToODEResidualAdapter(
            discretization, solver, row_treatment
        )
    if not isinstance(parameterization, ParameterizationProtocol):
        raise TypeError(
            f"parameterization must satisfy ParameterizationProtocol, "
            f"got {type(parameterization).__name__}"
        )
    derivs = parameterization.param_derivatives()
    if derivs.param_jacobian is not None and derivs.initial_param_jacobian is not None:
        if (
            derivs.param_param_hvp is not None
            and derivs.state_param_hvp is not None
            and derivs.param_state_hvp is not None
            and derivs.initial_param_hvp is not None
            and system.spatial_operator().state_derivatives().state_state_hvp
            is not None
        ):
            return GalerkinPhysicsToODEResidualWithHVPAdapter(
                discretization, solver, parameterization, row_treatment
            )
        return GalerkinPhysicsToODEResidualWithParamJacobianAdapter(
            discretization, solver, parameterization, row_treatment
        )
    return GalerkinPhysicsToODEResidualWithSetParamAdapter(
        discretization, solver, parameterization, row_treatment
    )
