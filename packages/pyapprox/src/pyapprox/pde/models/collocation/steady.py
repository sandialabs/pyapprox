"""Steady-state forward model for parameterized collocation PDEs.

Provides CollocationStateEquationWithJacobianAdapter (wraps
CollocationModel + parameterized physics as
ParameterizedStateEquationWithJacobianProtocol) and SteadyForwardModel
(satisfies FunctionProtocol with adjoint-based Jacobian computation).
"""

from typing import Any, Generic, Optional, Union

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.subset_of_states import (
    SubsetOfStatesAdjointFunctional,
)
from pyapprox.optimization.implicitfunction.operator.operator_with_jacobian import (
    AdjointOperatorWithJacobian,
)
from pyapprox.optimization.implicitfunction.operator.sensitivities import (
    VectorAdjointOperatorWithJacobian,
)
from pyapprox.pde.collocation.protocols.physics import (
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


class CollocationStateEquationWithJacobianAdapter(Generic[Array]):
    """Adapts CollocationModel for use with AdjointOperatorWithJacobian.

    Wraps CollocationModel + parameterized physics as
    ParameterizedStateEquationWithJacobianProtocol.

    Handles shape conversion between collocation convention (1D arrays)
    and protocol convention (2D column vectors):
        - State: collocation (nstates,) <-> protocol (nstates, 1)
        - Param: collocation (nparams,) <-> protocol (nparams, 1)
        - Residual return: (nstates, 1)
        - Jacobians: (nstates, nstates) and (nstates, nparams) -- same in both

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
        self._physics = model.physics()
        self._adapter = model.adapter()
        self._parameterization: ParameterizationProtocol[Array] = (
            parameterization
        )
        self._bc_indices = self._collect_bc_indices()

    def _collect_bc_indices(self) -> list[int]:
        """Collect all boundary DOF indices from physics BCs."""
        indices = []
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
                bc_idx = bc.boundary_indices()
                for ii in range(bc_idx.shape[0]):
                    indices.append(self._bkd.to_int(bc_idx[ii]))
        return indices

    def _zero_bc_rows(self, matrix: Array) -> Array:
        """Zero rows of a matrix at boundary DOF indices."""
        matrix = self._bkd.copy(matrix)
        for idx in self._bc_indices:
            if matrix.ndim == 1:
                matrix[idx] = 0.0
            else:
                matrix[idx, :] = 0.0
        return matrix

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
        self._adapter.set_time(0.0)
        residual = self._adapter(state_1d)
        jacobian = self._adapter.jacobian(state_1d)
        residual, _ = self._model._apply_boundary_conditions(
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
        self._adapter.set_time(0.0)
        residual = self._adapter(state_1d)
        jacobian = self._adapter.jacobian(state_1d)
        _, jacobian = self._model._apply_boundary_conditions(
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
        param_jac_fn = (
            self._parameterization.param_derivatives().param_jacobian
        )
        if param_jac_fn is None:
            raise RuntimeError(
                "parameterization does not provide param_jacobian"
            )
        pjac = param_jac_fn(state_1d, 0.0, param[:, 0])

        # Apply BC corrections (replaces _zero_bc_rows)
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
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
        self, bc: object, state_1d: Array, params_1d: Array, time: float
    ) -> object:
        """Build physical sensitivities dict for one BC's param_jacobian.

        Delegates d(flux·n)/dp computation to the parameterization via
        bc_flux_param_sensitivity. Only applies to BCs whose normal operator
        has coefficient dependence (e.g., flux Neumann with parameterized D).
        """
        bc_flux_fn = (
            self._parameterization.param_derivatives().bc_flux_param_sensitivity
        )
        if bc_flux_fn is None:
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
        dflux_n_dp = bc_flux_fn(state_1d, time, params_1d, bc_idx, normals)
        return {"dflux_n_dp": dflux_n_dp}


class CollocationStateEquationWithHVPAdapter(
    CollocationStateEquationWithJacobianAdapter[Array], Generic[Array]
):
    """Extends the steady adapter with the four HVP methods
    (ParameterizedStateEquationWithJacobianAndHVPProtocol).

    Collocation replaces ALL boundary rows (row_replaced superset of
    essential), so every second-derivative contraction receives the
    adjoint with ALL BC-row entries zeroed — the true-tensor
    convention: replaced rows must be affine in (state, params) jointly
    for their true second derivatives to vanish. That holds for
    Dirichlet and parameter-independent Neumann/Robin rows; BCs whose
    normal operator has coefficient dependence (parameterized-flux
    Neumann) carry nonzero second-order row terms this adapter cannot
    represent, so they are rejected at construction. State-shaped
    outputs are NOT masked (their BC-row entries are genuine and feed
    only the decoupled second-adjoint component).

    Requires physics satisfying PhysicsWithStateStateHVPProtocol and a
    parameterization with a second-order ParamDerivatives bundle.
    """

    def __init__(
        self,
        model: CollocationModel[Array],
        bkd: Backend[Array],
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ):
        super().__init__(model, bkd, parameterization=parameterization)
        if not isinstance(self._physics, PhysicsWithStateStateHVPProtocol):
            raise TypeError(
                "physics must satisfy PhysicsWithStateStateHVPProtocol "
                f"for the HVP tier, got {type(self._physics).__name__}"
            )
        self._hvp_physics: PhysicsWithStateStateHVPProtocol[Array] = (
            self._physics
        )
        derivs = self._parameterization.param_derivatives()
        if (
            derivs.param_param_hvp is None
            or derivs.state_param_hvp is None
            or derivs.param_state_hvp is None
        ):
            raise TypeError(
                "parameterization must provide a second-order "
                "ParamDerivatives bundle (all three parameter-facing "
                "HVPs); this adapter is the HVP tier"
            )
        self._param_param_hvp_fn: ParamHVPFn[Array] = derivs.param_param_hvp
        self._state_param_hvp_fn: ParamHVPFn[Array] = derivs.state_param_hvp
        self._param_state_hvp_fn: ParamHVPFn[Array] = derivs.param_state_hvp
        if hasattr(self._physics, "boundary_conditions"):
            for bc in self._physics.boundary_conditions():
                if not hasattr(bc, "normal_operator"):
                    continue
                normal_op = bc.normal_operator()
                if hasattr(
                    normal_op, "has_coefficient_dependence"
                ) and normal_op.has_coefficient_dependence():
                    raise NotImplementedError(
                        "HVP with coefficient-dependent BC rows "
                        "(parameterized-flux Neumann) is unsupported: "
                        "their second-order row sensitivities are not "
                        "representable by this adapter"
                    )

    def _zeroed_adjoint(self, adj_state: Array) -> Array:
        """Adjoint column as 1D with ALL BC-row entries zeroed."""
        adj = self._bkd.copy(adj_state[:, 0])
        for idx in self._bc_indices:
            adj[idx] = 0.0
        return adj

    def state_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/du^2) w with the BC-row adjoint zeroed.

        Shape: (nstates, 1).
        """
        self._set_param(param)
        return self._hvp_physics.state_state_hvp(
            state[:, 0], self._zeroed_adjoint(adj_state), wvec[:, 0], 0.0
        )[:, None]

    def param_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/dp^2) v with the BC-row adjoint zeroed.

        Shape: (nparams, 1).
        """
        self._set_param(param)
        return self._param_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def state_param_hvp(
        self, state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        """adj^T (d^2R/du dp) v with the BC-row adjoint zeroed.

        Shape: (nstates, 1).
        """
        self._set_param(param)
        return self._state_param_hvp_fn(
            state[:, 0],
            0.0,
            param[:, 0],
            self._zeroed_adjoint(adj_state),
            vvec[:, 0],
        )[:, None]

    def param_state_hvp(
        self, state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        """adj^T (d^2R/dp du) w with the BC-row adjoint zeroed.

        Shape: (nparams, 1).
        """
        self._set_param(param)
        return self._param_state_hvp_fn(
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
    Jacobian computation.

    Parameters
    ----------
    physics : object
        Collocation physics (must satisfy PhysicsProtocol). Parameter
        handling comes from the required ``parameterization``.
    bkd : Backend
        Computational backend.
    init_state : Array
        Initial guess for steady-state Newton solver. Shape: (nstates,).
    functional : ParameterizedFunctionalWithJacobianProtocol, optional
        QoI functional. If None, uses SubsetOfStatesAdjointFunctional with
        all indices (nqoi = nstates, identity functional).
    """

    def __init__(
        self,
        physics: Any,
        bkd: Backend[Array],
        init_state: Array,
        functional: Optional[ParameterizedFunctionalWithJacobianProtocol[Array]] = None,
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        self._bkd = bkd
        self._physics = physics
        self._parameterization: ParameterizationProtocol[Array] = (
            parameterization
        )
        self._init_state_1d = init_state
        self._init_state_2d = init_state[:, None]

        model = create_collocation_model(
            physics, bkd, parameterization=parameterization
        )
        self._state_eq = CollocationStateEquationWithJacobianAdapter(
            model, bkd, parameterization=parameterization
        )

        nstates = self._state_eq.nstates()
        nparams = self._state_eq.nparams()

        if functional is None:
            functional = SubsetOfStatesAdjointFunctional(
                nstates, nparams, bkd.arange(nstates), bkd
            )
        self._functional = functional

        # Lazy adjoint -- built on first jacobian call
        self._adjoint_op: Optional[
            Union[
                AdjointOperatorWithJacobian[Array],
                VectorAdjointOperatorWithJacobian[Array],
            ]
        ] = None
        self._nparams = nparams

        # Capability: the adjoint jacobian needs a parameter jacobian
        # from the parameterization's bundle
        self._has_param_jac = (
            parameterization.param_derivatives().param_jacobian is not None
        )
        if self._has_param_jac:
            self._derivs: Derivatives[Array] = Derivatives.first_order(
                jacobian=self._jacobian
            )
        else:
            self._derivs = Derivatives.none()

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

    def state_equation(self) -> CollocationStateEquationWithJacobianAdapter[Array]:
        """Return the state equation adapter."""
        return self._state_eq

    def adjoint_operator(self) -> object:
        """Return the adjoint operator (lazy-built on first access).

        Returns None if jacobian is not supported.
        """
        self._ensure_adjoint_op()
        return self._adjoint_op

    def _ensure_adjoint_op(self) -> None:
        """Build adjoint operator on first call."""
        if self._adjoint_op is not None:
            return
        if not self._has_param_jac:
            return
        if self._functional.nqoi() == 1:
            self._adjoint_op = AdjointOperatorWithJacobian(
                self._state_eq, self._functional
            )
        else:
            self._adjoint_op = VectorAdjointOperatorWithJacobian(
                self._state_eq, self._functional
            )

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
        self._ensure_adjoint_op()
        adjoint_op = self._adjoint_op
        if adjoint_op is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        return adjoint_op.jacobian(self._init_state_2d, sample)
