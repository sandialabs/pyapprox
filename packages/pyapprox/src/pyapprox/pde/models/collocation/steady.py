"""Steady-state forward model for parameterized collocation PDEs.

Provides CollocationParameterizedStateEquation (wraps
CollocationModel + parameterized physics as
SteadyStateEquationProtocol) and SteadyForwardModel
(satisfies FunctionProtocol with adjoint-based Jacobian computation).
"""

import warnings
from typing import Generic, Optional, Union

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesFunctional,
)
from pyapprox.optimization.implicitfunction.operator.adjoint_operator import (
    AdjointOperator,
)
from pyapprox.optimization.implicitfunction.operator.derivatives import (
    OperatorHVPFn,
)
from pyapprox.optimization.implicitfunction.operator.sensitivities import (
    ForwardSensitivityOperator,
)
from pyapprox.optimization.implicitfunction.state_equations.derivatives import (
    SteadyStateEquationDerivatives,
)
from pyapprox.pde.collocation.protocols.boundary import (
    BCPhysicalSensitivities,
    BoundaryConditionProtocol,
    BoundaryConditionWithNormalOperatorProtocol,
    BoundaryConditionWithParamJacobianProtocol,
    NormalOperatorProtocol,
)
from pyapprox.pde.collocation.protocols.physics import (
    PhysicsProtocol,
    PhysicsWithStateStateHVPProtocol,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.models.collocation.factory import (
    create_collocation_model,
)
from pyapprox.pde.parameterizations.derivatives import ParamHVPFn
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def _has_coefficient_dependent_bc_rows(
    physics: PhysicsProtocol[Array],
) -> bool:
    """Whether any BC row's normal term depends on a parameterized
    coefficient (parameterized-flux Neumann/Robin, hyperelastic
    traction).

    Such rows carry second-order sensitivities the HVP tier cannot
    represent — the single predicate shared by the HVP adapter's
    construction guard and the forward model's tier selection.
    """
    for bc in physics.boundary_conditions():
        if not isinstance(bc, BoundaryConditionWithNormalOperatorProtocol):
            continue
        if bc.normal_operator().has_coefficient_dependence():
            return True
    return False


class CollocationParameterizedStateEquation(Generic[Array]):
    """Adapts CollocationModel for use with AdjointOperator.

    Wraps CollocationModel + parameterized physics as
    SteadyStateEquationProtocol.

    Handles shape conversion between collocation convention (1D arrays)
    and protocol convention (2D column vectors):
        - State: collocation (nstates,) <-> protocol (nstates, 1)
        - Param: collocation (nparams,) <-> protocol (nparams, 1)
        - Residual return: (nstates, 1)
        - Jacobians: (nstates, nstates) and (nstates, nparams) -- same in both

    The ``derivatives()`` bundle carries the four adjoint-weighted HVPs
    when the physics satisfies PhysicsWithStateStateHVPProtocol, the
    parameterization's ParamDerivatives bundle is second order, and no
    BC row depends on a parameterized coefficient; otherwise it is
    empty.

    Collocation replaces ALL boundary rows (row_replaced superset of
    essential), so every second-derivative contraction receives the
    adjoint with ALL BC-row entries zeroed — the true-tensor
    convention: replaced rows must be affine in (state, params) jointly
    for their true second derivatives to vanish. That holds for
    Dirichlet and parameter-independent Neumann/Robin rows; BCs whose
    normal operator has coefficient dependence (parameterized-flux
    Neumann) carry nonzero second-order row terms this adapter cannot
    represent, so they withhold the HVPs. State-shaped outputs are NOT
    masked (their BC-row entries are genuine and feed only the
    decoupled second-adjoint component).

    Parameters
    ----------
    model : CollocationModel
        Collocation model wrapping the physics.
    bkd : Backend
        Computational backend.
    parameterization : ParameterizationProtocol
        Required; maps parameter vectors to physics coefficients and
        provides derivative capability via its ParamDerivatives bundle.
    """

    def __init__(
        self,
        model: CollocationModel[Array],
        bkd: Backend[Array],
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._model = model
        self._bkd = bkd
        physics = model.physics()
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"model physics must satisfy PhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        self._physics: PhysicsProtocol[Array] = physics
        self._adapter = model.adapter()
        self._parameterization: ParameterizationProtocol[Array] = parameterization
        self._bc_indices = self._collect_bc_indices()
        # Capability is fixed at construction: the narrowed second-order
        # callables are captured once, and None when any is absent.
        self._hvp_physics: Optional[PhysicsWithStateStateHVPProtocol[Array]] = None
        self._param_param_hvp_fn: Optional[ParamHVPFn[Array]] = None
        self._state_param_hvp_fn: Optional[ParamHVPFn[Array]] = None
        self._param_state_hvp_fn: Optional[ParamHVPFn[Array]] = None
        self._derivs: SteadyStateEquationDerivatives[Array] = (
            SteadyStateEquationDerivatives.none()
        )
        derivs = parameterization.param_derivatives()
        if (
            isinstance(physics, PhysicsWithStateStateHVPProtocol)
            and derivs.param_param_hvp is not None
            and derivs.state_param_hvp is not None
            and derivs.param_state_hvp is not None
            and not _has_coefficient_dependent_bc_rows(physics)
        ):
            self._hvp_physics = physics
            self._param_param_hvp_fn = derivs.param_param_hvp
            self._state_param_hvp_fn = derivs.state_param_hvp
            self._param_state_hvp_fn = derivs.param_state_hvp
            self._derivs = SteadyStateEquationDerivatives.second_order(
                self._state_state_hvp,
                self._state_param_hvp,
                self._param_state_hvp,
                self._param_param_hvp,
            )

    def _collect_bc_indices(self) -> list[int]:
        """Collect every residual row the physics BCs replace."""
        indices = []
        for bc in self._physics.boundary_conditions():
            bc_idx = bc.replaced_indices()
            for ii in range(bc_idx.shape[0]):
                indices.append(self._bkd.to_int(bc_idx[ii]))
        return indices

    def _set_param(self, param: Array) -> None:
        """Set parameter on physics (converts 2D column to 1D)."""
        self._parameterization.apply(param[:, 0])

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return number of state variables."""
        return self._model.nstates()

    def nparams(self) -> int:
        """Return number of parameters."""
        return self._parameterization.nparams()

    def derivatives(self) -> SteadyStateEquationDerivatives[Array]:
        """Return the optional derivatives this state equation provides."""
        return self._derivs

    def solve(self, init_state: Array, param: Array) -> Array:
        """Solve the steady-state problem R(u, p) = 0 for u.

        Parameters
        ----------
        init_state : Array
            Initial guess for Newton solver. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Solution state. Shape: (nstates, 1).
        """
        self._set_param(param)
        sol_1d = self._model.solve_steady(init_state[:, 0])
        return sol_1d[:, None]

    def __call__(self, state: Array, param: Array) -> Array:
        """Compute the residual R(u, p) with boundary conditions applied.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Residual vector. Shape: (nstates, 1).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        # Steady problem evaluated at t = 0: collocation physics have no
        # time-free view yet, so time-dependent data is read at t = 0.
        residual = self._adapter(state_1d, 0.0)
        jacobian = self._adapter.jacobian(state_1d, 0.0)
        residual, _ = self._physics.apply_boundary_conditions(
            residual, jacobian, state_1d, 0.0
        )
        return residual[:, None]

    def state_jacobian(self, state: Array, param: Array) -> Array:
        """Compute Jacobian of residual w.r.t. state, dR/du.

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            State Jacobian. Shape: (nstates, nstates).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        # Steady problem evaluated at t = 0 (see ``residual``).
        residual = self._adapter(state_1d, 0.0)
        jacobian = self._adapter.jacobian(state_1d, 0.0)
        _, jacobian = self._physics.apply_boundary_conditions(
            residual, jacobian, state_1d, 0.0
        )
        return jacobian

    def param_jacobian(self, state: Array, param: Array) -> Array:
        """Compute Jacobian of residual w.r.t. parameters, dR/dp.

        Applies each BC's apply_to_param_jacobian with physical sensitivities
        (replaces blanket zero-rows approach to support coefficient-dependent
        BCs like flux Neumann with parameterized diffusivity).

        Parameters
        ----------
        state : Array
            State vector. Shape: (nstates, 1).
        param : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        Array
            Parameter Jacobian. Shape: (nstates, nparams).
        """
        self._set_param(param)
        state_1d = state[:, 0]
        param_jac_fn = self._parameterization.param_derivatives().param_jacobian
        if param_jac_fn is None:
            raise RuntimeError("parameterization does not provide param_jacobian")
        pjac = param_jac_fn(state_1d, 0.0, param[:, 0])

        # Apply each BC's parameter-Jacobian row correction
        for bc in self._physics.boundary_conditions():
            if not isinstance(bc, BoundaryConditionWithParamJacobianProtocol):
                raise TypeError(
                    f"BC {type(bc).__name__} must satisfy "
                    f"BoundaryConditionWithParamJacobianProtocol "
                    f"for parameter sensitivity"
                )
            phys_sens = self._build_bc_physical_sensitivities(
                bc, state_1d, param[:, 0], 0.0
            )
            pjac = bc.apply_to_param_jacobian(
                pjac,
                state_1d,
                0.0,
                physical_sensitivities=phys_sens,
            )
        return pjac

    def _build_bc_physical_sensitivities(
        self,
        bc: BoundaryConditionProtocol[Array],
        state_1d: Array,
        params_1d: Array,
        time: float,
    ) -> Optional[BCPhysicalSensitivities[Array]]:
        """Build physical sensitivities for one BC's param_jacobian.

        Delegates d(flux·n)/dp computation to the parameterization via
        bc_flux_param_sensitivity. Only applies to BCs whose normal operator
        has coefficient dependence (e.g., flux Neumann with parameterized D).
        """
        bc_flux_fn = (
            self._parameterization.param_derivatives().bc_flux_param_sensitivity
        )
        if bc_flux_fn is None:
            return None
        if not isinstance(bc, BoundaryConditionWithNormalOperatorProtocol):
            return None
        normal_op: NormalOperatorProtocol[Array] = bc.normal_operator()
        if not normal_op.has_coefficient_dependence():
            return None
        bc_idx = bc.boundary_indices()
        normals = normal_op.normals()
        dflux_n_dp = bc_flux_fn(state_1d, time, params_1d, bc_idx, normals)
        return BCPhysicalSensitivities(dflux_n_dp=dflux_n_dp)

    def _zeroed_adjoint(self, adj_state: Array) -> Array:
        """Adjoint column as 1D with ALL BC-row entries zeroed."""
        adj = self._bkd.copy(adj_state[:, 0])
        for idx in self._bc_indices:
            adj[idx] = 0.0
        return adj

    @staticmethod
    def _unavailable() -> RuntimeError:
        return RuntimeError(
            "the HVPs are unavailable; check derivatives() before calling"
        )

    def _state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/du^2) w with the BC-row adjoint zeroed.

        Shape: (nstates, 1).
        """
        hvp_physics = self._hvp_physics
        if hvp_physics is None:
            raise self._unavailable()
        self._set_param(param)
        return hvp_physics.state_state_hvp(
            state[:, 0], self._zeroed_adjoint(adj_state), wvec[:, 0], 0.0
        )[:, None]

    def _param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/dp^2) v with the BC-row adjoint zeroed.

        Shape: (nparams, 1).
        """
        param_param_hvp_fn = self._param_param_hvp_fn
        if param_param_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return param_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def _state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/du dp) v with the BC-row adjoint zeroed.

        Shape: (nstates, 1).
        """
        state_param_hvp_fn = self._state_param_hvp_fn
        if state_param_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return state_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def _param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/dp du) w with the BC-row adjoint zeroed.

        Shape: (nparams, 1).
        """
        param_state_hvp_fn = self._param_state_hvp_fn
        if param_state_hvp_fn is None:
            raise self._unavailable()
        self._set_param(param)
        return param_state_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            wvec[:, 0],
        )[:, None]


class SteadyForwardModel(Generic[Array]):
    """Steady-state parameterized PDE forward model.

    Maps PDE parameters to quantities of interest extracted from the
    steady-state solution. Satisfies FunctionProtocol with adjoint-based
    Jacobian computation and, when the parameterization bundle is
    second order, the physics provides its state-state contraction, no
    BC row depends on a parameterized coefficient, and the QoI is
    scalar, a second-order-adjoint HVP (the tier is fixed at
    construction; an otherwise-eligible downgrade warns).

    Parameters
    ----------
    physics : PhysicsProtocol
        Collocation physics. Parameter handling comes from the required
        ``parameterization``.
    bkd : Backend
        Computational backend.
    init_state : Array
        Initial guess for steady-state Newton solver. Shape: (nstates,).
    functional : SteadyFunctionalProtocol, optional
        QoI functional. If None, uses SubsetOfStatesFunctional with
        all indices (nqoi = nstates, identity functional).
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        init_state: Array,
        functional: Optional[SteadyFunctionalProtocol[Array]] = None,
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"physics must satisfy PhysicsProtocol, got {type(physics).__name__}"
            )
        self._bkd = bkd
        self._physics = physics
        self._parameterization: ParameterizationProtocol[Array] = parameterization
        self._init_state_1d = init_state
        self._init_state_2d = init_state[:, None]

        model = create_collocation_model(
            physics, bkd, parameterization=parameterization
        )
        nstates = model.nstates()
        nparams = parameterization.nparams()
        self._nparams = nparams

        if functional is None:
            functional = SubsetOfStatesFunctional(
                nstates, nparams, bkd.arange(nstates), bkd
            )
        # A transient functional has every member of the steady
        # protocol, so isinstance cannot reject it; the declared time
        # dependence does.
        if functional.time_dependence() != "steady":
            raise TypeError(
                f"{type(functional).__name__} is a "
                f"{functional.time_dependence()} functional; a steady "
                "model needs one that reads a single state"
            )
        self._functional = functional

        # Construction-time capability selection, read from bundles. A
        # functional without Jacobians is evaluated, not differentiated.
        # A scalar functional is differentiated by the adjoint method,
        # whose HVP exists when the state equation and the functional
        # both carry their second derivatives; a vector functional by
        # forward sensitivities.
        bundle = parameterization.param_derivatives()
        q_derivs = functional.derivatives()
        self._state_eq = CollocationParameterizedStateEquation(
            model, bkd, parameterization=parameterization
        )
        self._jacobian_op: Optional[
            Union[AdjointOperator[Array], ForwardSensitivityOperator[Array]]
        ] = None
        self._operator_hvp: Optional[OperatorHVPFn[Array]] = None
        self._derivs: Derivatives[Array] = Derivatives.none()
        if bundle.param_jacobian is None or q_derivs.state_jacobian is None:
            return
        if functional.nqoi() != 1:
            self._jacobian_op = ForwardSensitivityOperator(self._state_eq, functional)
            self._derivs = Derivatives.first_order(jacobian=self._jacobian)
            return
        adjoint_op = AdjointOperator(self._state_eq, functional)
        self._jacobian_op = adjoint_op
        operator_hvp = adjoint_op.derivatives().hvp
        if operator_hvp is not None:
            self._operator_hvp = operator_hvp
            self._derivs = Derivatives.second_order(
                jacobian=self._jacobian, hvp=self._hvp
            )
            return
        # The functional bundle carries its four HVPs together, so one
        # field stands for all of them.
        if (
            bundle.param_param_hvp is not None
            and bundle.state_param_hvp is not None
            and bundle.param_state_hvp is not None
            and q_derivs.state_state_hvp is not None
        ):
            warnings.warn(
                "second-order parameterization bundle downgraded to "
                "the Jacobian tier: the physics lacks state_state_hvp "
                "or a BC row depends on a parameterized coefficient",
                UserWarning,
                stacklevel=2,
            )
        self._derivs = Derivatives.first_order(jacobian=self._jacobian)

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle (jacobian w.r.t. parameters)."""
        return self._derivs

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of input variables (parameters)."""
        return self._state_eq.nparams()

    def nqoi(self) -> int:
        """Return number of output quantities of interest."""
        return self._functional.nqoi()

    def state_equation(self) -> CollocationParameterizedStateEquation[Array]:
        """Return the state equation adapter."""
        return self._state_eq

    def __call__(self, samples: Array) -> Array:
        """Evaluate forward model for multiple parameter samples.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nvars, nsamples).

        Returns
        -------
        Array
            QoI values. Shape: (nqoi, nsamples).
        """
        bkd = self._bkd
        nsamples = samples.shape[1]
        nqoi = self.nqoi()
        result = bkd.zeros((nqoi, nsamples))
        for ii in range(nsamples):
            param = samples[:, ii : ii + 1]
            sol = self._state_eq.solve(self._init_state_2d, param)
            qoi = self._functional(sol, param)
            if qoi.ndim == 2:
                result[:, ii : ii + 1] = qoi
            else:
                result[:, ii] = qoi
        return result

    def _jacobian(self, sample: Array) -> Array:
        """Compute Jacobian of QoI w.r.t. parameters using adjoint method.

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nqoi, nvars).
        """
        jacobian_op = self._jacobian_op
        if jacobian_op is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        return jacobian_op.jacobian(self._init_state_2d, sample)

    def _hvp(self, sample: Array, vvec: Array) -> Array:
        """Compute (d^2Q/dp^2) v via the second-order adjoint.

        Parameters
        ----------
        sample : Array
            Single parameter sample. Shape: (nvars, 1).
        vvec : Array
            Direction vector. Shape: (nvars, 1).

        Returns
        -------
        Array
            HVP result. Shape: (nvars, 1).
        """
        operator_hvp = self._operator_hvp
        if operator_hvp is None:
            raise RuntimeError("hvp is unavailable; check derivatives() before calling")
        return operator_hvp(self._init_state_2d, sample, vvec)
