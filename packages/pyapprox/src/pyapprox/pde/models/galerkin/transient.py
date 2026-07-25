"""Transient forward model for parameterized galerkin PDEs.

``GalerkinTransientForwardModel`` maps PDE parameters to quantities of
interest extracted from the transient solution. It satisfies
``FunctionProtocol``: the Jacobian comes from the adjoint method
(scalar QoI, reusing the already-computed trajectory) or the shared
tangent-linear sweep (all-states QoI), and scalar QoIs additionally
expose a Hessian-vector product through the second-order adjoint when
the parameterization's bundle supports it.

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
from pyapprox.ode.operator.forward_sensitivity import (
    solve_final_forward_sensitivity,
)
from pyapprox.ode.operator.time_adjoint_hvp import (
    TimeAdjointOperatorWithHVP,
)
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinPhysicsProtocol,
)
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.pde.models.galerkin.physics_adapter import (
    GalerkinPhysicsToODEResidualWithHVPAdapter,
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend

_TransientFunctional = Union[
    TransientFunctionalWithJacobianProtocol[Array],
    TransientFunctionalWithJacobianAndHVPProtocol[Array],
]


class GalerkinTransientForwardModel(GalerkinModel[Array]):
    """Transient parameterized galerkin PDE forward model.

    Satisfies ``FunctionProtocol``: ``__call__`` maps parameter samples
    to QoI values; ``derivatives()`` carries the adjoint Jacobian and,
    for scalar QoIs with a second-order parameterization bundle, the
    second-order-adjoint HVP. Capability is decided once at
    construction from the parameterization's ``ParamDerivatives``
    bundle and the functional's protocol tier — never by ``hasattr``.

    Extends :class:`GalerkinModel`, so the plain ``solve_steady`` /
    ``solve_transient`` surface remains available.

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        Galerkin physics. Each parameterized coefficient must be in its
        differentiable representation (validated by the
        parameterization's facade at its construction).
    parameterization : ParameterizationProtocol
        Maps parameter vectors to physics coefficients; must be bound
        to the same physics instance.
    init_state : Array
        Initial condition for the transient solve. Shape: (nstates,).
    time_config : TimeIntegrationConfig
        Time integration configuration.
    bkd : Backend
        Computational backend.
    functional : transient functional, optional
        QoI functional. Defaults to ``AllStatesEndpointFunctional``
        (nqoi = nstates: the full final-time state).
    """

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        parameterization: ParameterizationProtocol[Array],
        init_state: Array,
        time_config: TimeIntegrationConfig[Array],
        bkd: Backend[Array],
        functional: Optional[_TransientFunctional[Array]] = None,
    ) -> None:
        if not isinstance(parameterization, ParameterizationProtocol):
            raise TypeError(
                "parameterization must satisfy ParameterizationProtocol, "
                f"got {type(parameterization).__name__}"
            )
        if parameterization.physics() is not physics:
            raise ValueError(
                "parameterization is bound to a different physics "
                "instance than the one passed to the model"
            )
        adapter = create_galerkin_physics_ode_residual(
            physics, parameterization
        )
        super().__init__(physics, bkd, adapter=adapter)
        self._parameterization = parameterization
        # Inject Dirichlet values ONCE so every consumer of the initial
        # state (solve_transient re-injects idempotently; the HVP
        # operator's internal forward solve does not inject) sees the
        # same constrained state.
        self._init_state = physics.constraint_set().inject(
            init_state, time_config.init_time
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
                self.adapter(), GalerkinPhysicsToODEResidualWithHVPAdapter
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
    ) -> GalerkinPhysicsToODEResidualWithParamJacobianAdapter[Array]:
        adapter = self.adapter()
        if not isinstance(
            adapter, GalerkinPhysicsToODEResidualWithParamJacobianAdapter
        ):
            raise TypeError(
                "derivative evaluation requires a param-jacobian adapter "
                f"tier; got {type(adapter).__name__}"
            )
        return adapter

    def _forward_solve(self, param_2d: Array) -> Tuple[Array, Array]:
        """Rebind parameters and solve the transient problem.

        Parameters
        ----------
        param_2d : Array
            Parameter vector. Shape: (nparams, 1).

        Returns
        -------
        solutions : Array
            Trajectory. Shape: (nstates, ntimes).
        times : Array
            Time points. Shape: (ntimes,).
        """
        param_1d = param_2d[:, 0]
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
            fwd_sols, _ = self._forward_solve(param_2d)
            qoi = self._functional(fwd_sols, param_2d)
            if qoi.ndim == 2:
                result[:, ii : ii + 1] = qoi
            else:
                result[:, ii] = qoi
        return result

    def _jacobian(self, sample: Array) -> Array:
        """Compute dQ/dp for one sample. Shape: (nqoi, nvars).

        Scalar QoI: adjoint sweep over the just-computed trajectory.
        All-states QoI: shared tangent-linear sweep for the full
        ``dy(T)/dp`` matrix. Other vector QoIs are not supported.
        """
        fwd_sols, times = self._forward_solve(sample)
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
        return solve_final_forward_sensitivity(
            integrator.time_residual(), fwd_sols, times, self._bkd
        )

    def _hvp(self, sample: Array, vvec: Array) -> Array:
        """Compute (d^2Q/dp^2) v via the second-order adjoint.

        Shape: (nvars, 1). The operator runs its own forward solve for
        the given parameters (its trajectory storage is cleared here
        because parameter rebinding invalidates it).
        """
        if self._hvp_functional is None:
            raise RuntimeError(
                "hvp is unavailable; check derivatives() before calling"
            )
        self._forward_solve(sample)
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
