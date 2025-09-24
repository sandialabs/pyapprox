from abc import abstractmethod

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.bayes.variational.flows import Flow
from pyapprox.variables.joint import JointVariable
from pyapprox.util.newton import NewtonSolver
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    TimeIntegratorNewtonResidual,
    # HeunResidual,
    ForwardEulerResidual,
)
from pyapprox.surrogates.affine.basisexp import BasisExpansion


class VelocityModel(TransientNewtonResidual):
    def __init__(self, nstates: int, backend: BackendMixin):
        super().__init__(backend)
        self._nstates = nstates

    def _nstates(self) -> int:
        return self._nstates

    def set_time(self, time: float):
        self._time = time

    @abstractmethod
    def _value(self, state: Array) -> Array:
        raise NotImplementedError

    def __call__(self, state: Array) -> Array:
        if state.ndim != 1:
            raise ValueError("state.ndim must equal 1")
        values = self._value(state)
        if values.ndim != 1:
            raise ValueError(
                f"self._value must return 1D array with shape {state.shape}"
            )
        return values


class BasisExpansionVelocityModel(VelocityModel):
    def __init__(self, bexp: BasisExpansion):
        if not isinstance(bexp, BasisExpansion):
            raise ValueError("bexp must be an instance of BasisExpansion")
        # first variable that bexp accepts is time so nstatates is nvars-1
        if bexp.nvars() != bexp.nqoi() + 1:
            raise ValueError("bexp.nvars() != bexp.nqoi()+1 are inconsistent")
        super().__init__(bexp.nqoi(), bexp._bkd)
        self._bexp = bexp

    def _value(self, state: Array) -> Array:
        return self._bexp(
            self._bkd.hstack((self._bkd.asarray(self._time)[None], state))[
                :, None
            ]
        )[0]


class FlowODEModel:
    def __init__(
        self,
        time_residual: TimeIntegratorNewtonResidual,
        deltat: float,
        newton_solver: NewtonSolver = None,
    ):
        self._time_int = ImplicitTimeIntegrator(
            time_residual,
            0.0,
            1.0,
            deltat,
            newton_solver=newton_solver,
            verbosity=0,
        )

    def __call__(self, source_sample: Array) -> Array:
        result = self._time_int.solve(source_sample)
        return result[0][:, -1]


class ContinuousNormalizingFlow(Flow):
    def __init__(
        self,
        source_variable: JointVariable,
        vel_model: VelocityModel,
        deltat: float,
        nlabels: int = 0,
        time_residual_cls: TimeIntegratorNewtonResidual = ForwardEulerResidual,
    ):
        super().__init__(source_variable)
        if not isinstance(vel_model, VelocityModel):
            raise ValueError("vel_model must be an instance of VelocityModel")
        self._vel_model = vel_model
        if not self._bkd.bkd_equal(self._bkd, vel_model._bkd):
            raise ValueError(
                "backend of joint variable and vel_model must be the same"
            )
        self._ode_model = FlowODEModel(
            time_residual_cls(self._vel_model), deltat
        )
        self._nlabels = nlabels

    def nlabels(self) -> int:
        return self._nlabels

    def _map_from_latent_single_sample(self, latent_sample: Array):
        # Solve the ODE
        return self._ode_model(latent_sample)

    def _map_from_latent_many_sample(self, usamples: Array):
        results = [
            self._map_from_latent_single_sample(usample)
            for usample in usamples.T
        ]
        return self._bkd.stack(results, axis=1)

    def _map_to_latent_many_sample(self, usamples: Array):
        results = [
            self._map_to_latent_single_sample(usample)
            for usample in usamples.T
        ]
        # samples = self._bkd.asarray([result[0] for result in results])
        logpdf_vals = self._bkd.asarray([result[0] for result in results])
        return logpdf_vals

    def _map_to_latent(self, usamples: Array) -> Array:
        return self._map_to_latent_many_sample(usamples)[0]

    def _map_from_latent(self, usamples: Array) -> Array:
        return self._map_from_latent_many_sample(usamples)

    def logpdf(self, samples: Array) -> Array:
        return self._map_to_latent_many_sample(samples)[1]
