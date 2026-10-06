"""Transient forward model for parameterized collocation PDEs.

``TransientForwardModel`` maps PDE parameters to quantities of
interest extracted from the transient solution. It satisfies
``FunctionProtocol``: the Jacobian comes from the caller's choice of
the adjoint method or the tangent-linear sweep, applied to the
already-computed trajectory; scalar QoIs additionally expose
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
from pyapprox.ode.functionals.protocols import (
    TransientFunctionalWithJacobianAndHVPProtocol,
    TransientFunctionalWithJacobianProtocol,
)
from pyapprox.ode.operator.qoi_jacobian import (
    TransientQoIJacobianMethod,
    default_qoi_jacobian_method,
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
from pyapprox.pde.parameterizations.binding import require_owned_targets
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
    jacobian_method : TransientQoIJacobianMethod, optional
        How ``derivatives().jacobian`` computes dQ/dp: ``adjoint_jacobian``
        (one backward sweep per QoI) or ``forward_sensitivity_jacobian``
        (one tangent-linear sweep with a column per parameter), both in
        ``pyapprox.ode.operator``. The functional must support the
        chosen method. Default: ``adjoint_jacobian`` when nqoi = 1,
        ``forward_sensitivity_jacobian`` otherwise.
    """

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        bkd: Backend[Array],
        init_state: Array,
        time_config: TimeIntegrationConfig[Array],
        functional: Optional[_TransientFunctional[Array]] = None,
        parameterization: Optional[ParameterizationProtocol[Array]] = None,
        jacobian_method: Optional[TransientQoIJacobianMethod[Array]] = None,
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
        require_owned_targets(parameterization, physics)
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
        self._jacobian_method = default_qoi_jacobian_method(
            functional, jacobian_method
        )

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

        Applies the constructor's ``jacobian_method`` to the
        just-computed trajectory.
        """
        fwd_sols, times = self.forward_solve(sample)
        return self._jacobian_method(
            self.last_integrator(), self._functional, fwd_sols, times, sample
        )

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
