"""Transient forward model for parameterized collocation PDEs.

``TransientForwardModel`` maps PDE parameters to quantities of
interest extracted from the transient solution. It satisfies
``FunctionProtocol``: the Jacobian comes from the adjoint method
(scalar QoI, reusing the already-computed trajectory) or, for the
all-states QoI, whichever of the shared tangent-linear sweep and the
row-wise adjoint costs fewer sweeps; scalar QoIs additionally expose
a Hessian-vector product through the second-order adjoint when the
parameterization's bundle supports it.

The whole pipeline (parameterized adapter tier, model) is constructed
once; per evaluation only the parameter values are rebound.
"""

from typing import Optional, Tuple, Union

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.functionals.all_states_endpoint import (
    AllStatesEndpointFunctional,
)
from pyapprox.ode.functionals.endpoint import EndpointFunctional
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.operator.forward_sensitivity import (
    solve_final_forward_sensitivity,
)
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.pde.collocation.protocols.boundary import (
    DirichletBCProtocol,
)
from pyapprox.pde.collocation.protocols.physics import (
    PhysicsProtocol,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.models.collocation.physics_adapter import (
    CollocationPhysicsToODEResidualWithHVPAdapter,
    CollocationPhysicsToODEResidualWithSetParamAdapter,
    create_collocation_physics_ode_residual,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

_TransientFunctional = Union[
    TransientFunctionalWithJacobianProtocol[Array],
    TransientFunctionalWithJacobianAndHVPProtocol[Array],
]


class TransientForwardModel(CollocationModel[Array]):
    """Transient parameterized collocation PDE forward model.

    Satisfies ``FunctionProtocol``: ``__call__`` maps parameter samples
    to QoI values; ``derivatives()`` carries the adjoint Jacobian and,
    for scalar QoIs with a second-order parameterization bundle, the
    second-order-adjoint HVP. Capability is decided once at
    construction from the parameterization's ``ParamDerivatives``
    bundle, the functional's protocol tier, and the adapter tier —
    never by ``hasattr``.

    Extends :class:`CollocationModel`, so the plain ``solve_steady`` /
    ``solve_transient`` surface remains available.

    Parameters
    ----------
    physics : PhysicsProtocol
        Collocation physics.
    bkd : Backend
        Computational backend.
    init_state : Array
        Initial condition. Shape: (nstates,). Essential (Dirichlet)
        values are injected once at construction so the trajectory,
        the adjoint machinery, and the HVP operator's internal forward
        solve all see the same constrained state.
    time_config : TimeIntegrationConfig
        Time integration configuration.
    functional : transient functional, optional
        QoI functional. Defaults to ``AllStatesEndpointFunctional``
        (nqoi = nstates: the full final-time state).
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients; must be bound
        to the same physics instance.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        init_state: Array,
        time_config: TimeIntegrationConfig[Array],
        functional: Optional[_TransientFunctional[Array]] = None,
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                f"parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        if not isinstance(physics, PhysicsProtocol):
            raise TypeError(
                f"physics must satisfy PhysicsProtocol, "
                f"got {type(physics).__name__}"
            )
        if parameterization.physics() is not physics:
            raise ValueError(
                "parameterization is bound to a different physics "
                "instance than the one passed to the model"
            )
        adapter = create_collocation_physics_ode_residual(
            physics, bkd, parameterization
        )
        super().__init__(physics, bkd, adapter=adapter)
        self._parameterization = parameterization
        self._init_state = self._inject_essential_values(
            physics, bkd, init_state, time_config.init_time
        )
        self._time_config = time_config
        self._nparams = parameterization.nparams()

        if functional is None:
            functional = AllStatesEndpointFunctional(
                physics.nstates(), self._nparams, bkd
            )
        if not isinstance(
            functional, TransientFunctionalWithJacobianProtocol
        ):
            raise TypeError(
                "functional must satisfy "
                "TransientFunctionalWithJacobianProtocol, got "
                f"{type(functional).__name__}"
            )
        self._functional: _TransientFunctional[Array] = functional

        bundle = parameterization.param_derivatives()
        self._hvp_functional: Optional[
            TransientFunctionalWithJacobianAndHVPProtocol[Array]
        ] = None
        if bundle.param_jacobian is None:
            self._derivs: Derivatives[Array] = Derivatives.none()
        elif (
            self._functional.nqoi() == 1
            and isinstance(
                self._functional,
                TransientFunctionalWithJacobianAndHVPProtocol,
            )
            and isinstance(
                self.adapter(),
                CollocationPhysicsToODEResidualWithHVPAdapter,
            )
        ):
            self._hvp_functional = self._functional
            self._derivs = Derivatives.second_order(
                jacobian=self._jacobian, hvp=self._hvp
            )
        else:
            self._derivs = Derivatives.first_order(
                jacobian=self._jacobian
            )

    @staticmethod
    def _inject_essential_values(
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        init_state: Array,
        init_time: float,
    ) -> Array:
        """Set essential (Dirichlet) values on the initial condition.

        A raw initial condition that violates the essential values
        feeds every derivative path a state the forward solve never
        produces; injecting once here keeps them consistent.
        """
        injected = bkd.copy(init_state)
        for bc in physics.boundary_conditions():
            if not bc.is_essential():
                continue
            if not isinstance(bc, DirichletBCProtocol):
                continue
            bc_idx = bc.boundary_indices()
            values = bc.boundary_values(init_time)
            for ii in range(bc_idx.shape[0]):
                injected[bkd.to_int(bc_idx[ii])] = values[ii]
        return injected

    def derivatives(self) -> Derivatives[Array]:
        """Return the derivative bundle (w.r.t. parameters)."""
        return self._derivs

    def nvars(self) -> int:
        """Return the number of input variables (parameters)."""
        return self._nparams

    def nqoi(self) -> int:
        """Return the number of output quantities of interest."""
        return self._functional.nqoi()

    def _param_adapter(
        self,
    ) -> CollocationPhysicsToODEResidualWithSetParamAdapter[Array]:
        adapter = self.adapter()
        if not isinstance(
            adapter, CollocationPhysicsToODEResidualWithSetParamAdapter
        ):
            raise TypeError(
                "derivative evaluation requires a parameterized adapter "
                f"tier; got {type(adapter).__name__}"
            )
        return adapter

    def forward_solve(self, sample: Array) -> Tuple[Array, Array]:
        """Rebind parameters and solve the transient problem.

        Returns the trajectory the QoI functional sees — for plotting
        and post-processing; ``__call__`` remains the QoI path.

        Parameters
        ----------
        sample : Array
            Parameter vector. Shape: (nvars, 1).

        Returns
        -------
        solutions : Array
            Trajectory. Shape: (nstates, ntimes).
        times : Array
            Time points. Shape: (ntimes,).
        """
        if sample.ndim != 2 or sample.shape[1] != 1:
            raise ValueError(
                f"sample must have shape (nvars, 1), got {sample.shape}"
            )
        param_1d = sample[:, 0]
        self._parameterization.apply(param_1d)
        self._param_adapter().set_param(param_1d)
        return self.solve_transient(self._init_state, self._time_config)

    def __call__(self, samples: Array) -> Array:
        """Evaluate the QoI for parameter samples.

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
        result = bkd.zeros((self.nqoi(), nsamples))
        result = bkd.copy(result)
        for ii in range(nsamples):
            param_2d = samples[:, ii : ii + 1]
            fwd_sols, _ = self.forward_solve(param_2d)
            qoi = self._functional(fwd_sols, param_2d)
            if qoi.ndim == 2:
                result[:, ii : ii + 1] = qoi
            else:
                result[:, ii] = qoi
        return result

    def _jacobian(self, sample: Array) -> Array:
        """Compute dQ/dp for one sample. Shape: (nqoi, nvars).

        Scalar QoI: adjoint sweep over the just-computed trajectory.
        All-states QoI: a costed choice — the tangent-linear sweep
        costs one linear solve per parameter, the row-wise adjoint one
        backward sweep per QoI, so the smaller of
        ``(nparams, nqoi)`` decides (a KLE-sized ``nparams`` must not
        silently pay the O(nparams) factor). Other vector QoIs are
        not supported.
        """
        fwd_sols, times = self.forward_solve(sample)
        integrator = self.last_integrator()
        if self._functional.nqoi() == 1:
            integrator.set_functional(self._functional)
            return integrator.gradient(fwd_sols, times, sample)
        if not isinstance(self._functional, AllStatesEndpointFunctional):
            raise NotImplementedError(
                "vector-QoI jacobians are only implemented for "
                "AllStatesEndpointFunctional (dQ/dy(T) = I); got "
                f"{type(self._functional).__name__}"
            )
        nqoi = self._functional.nqoi()
        if self._nparams <= nqoi:
            return solve_final_forward_sensitivity(
                integrator.time_residual(), fwd_sols, times, self._bkd
            )
        bkd = self._bkd
        result = bkd.copy(bkd.zeros((nqoi, self._nparams)))
        for k in range(nqoi):
            integrator.set_functional(
                EndpointFunctional(k, nqoi, self._nparams, bkd)
            )
            row = integrator.gradient(fwd_sols, times, sample)
            result[k, :] = row[0, :]
        return result

    def _hvp(self, sample: Array, vvec: Array) -> Array:
        """Compute (d^2Q/dp^2) v via the second-order adjoint.

        Shape: (nvars, 1). The operator runs its own forward solve for
        the given parameters (a fresh operator per call: its
        trajectory storage would be invalidated by the parameter
        rebinding).
        """
        if self._hvp_functional is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        self.forward_solve(sample)
        operator = TimeAdjointOperatorWithHVP(
            self.last_integrator(), self._hvp_functional
        )
        return operator.hvp(self._init_state, sample, vvec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"physics={self._physics.__class__.__name__}, "
            f"nqoi={self.nqoi()}, "
            f"nvars={self.nvars()})"
        )
