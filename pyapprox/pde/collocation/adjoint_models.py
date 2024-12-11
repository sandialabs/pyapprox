from abc import abstractmethod, ABC
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.interface.model import SingleSampleModel
from pyapprox.pde.collocation.newton import (
    NewtonResidual,
    NewtonSolver,
    AdjointFunctional,
    AdjointSolver,
)
from pyapprox.pde.collocation.timeintegration import (
    TransientAdjointFunctional,
    TimeIntegratorNewtonResidual,
    ImplicitTimeIntegrator,
)


class AdjointModel(SingleSampleModel, ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = backend.jacobian_implemented()

    def nqoi(self) -> int:
        return self._functional.nqoi()

    @abstractmethod
    def nvars(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _fwd_solve(self):
        raise NotImplementedError

    @abstractmethod
    def forward_solve(self, sample):
        raise NotImplementedError

    @abstractmethod
    def set_param(self, param: Array):
        raise NotImplementedError

    @abstractmethod
    def _eval_functional(self) -> Array:
        raise NotImplementedError

    def _evaluate(self, sample: Array) -> Array:
        self.set_param(sample[:, 0])
        self._fwd_solve()
        return self._eval_functional()

    @abstractmethod
    def _jacobian_from_adjoint(self) -> Array:
        raise NotImplementedError

    def _jacobian(self, sample: Array) -> Array:
        self.set_param(sample[:, 0])
        return self._jacobian_from_adjoint()

    def _apply_hessian_from_adjoint(self, vec: Array) -> Array:
        raise NotImplementedError("_hessian_from_adjoint not implemented")

    def _apply_hessian(self, sample: Array, vec: Array) -> Array:
        self.set_param(sample[:, 0])
        return self._apply_hessian_from_adjoint(vec[:, 0])


class SteadyAdjointModel(AdjointModel):
    def __init__(
        self,
        residual: NewtonResidual,
        functional: AdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        jacobian_mode: str = "backward",
    ):
        super().__init__(residual._bkd)
        self._residual = residual
        self._functional = functional
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self._newton_solver = newton_solver
        self._newton_solver.set_residual(self._residual)
        self._adjoint_solver = AdjointSolver(
            self._newton_solver, self._functional
        )
        self._jacobian_mode = jacobian_mode
        self._apply_hessian_implemented = self._bkd.hessian_implemented()

    def set_functional(self, functional: AdjointFunctional):
        self._functional = functional
        self._adjoint_solver.set_functional(self._functional)

    def set_param(self, param: Array):
        self._adjoint_solver.set_param(param)

    def _fwd_solve(self):
        self._sols = self._adjoint_solver.forward_solve()

    def forward_solve(self, sample) -> Array:
        self.set_param(sample[:, 0])
        self._fwd_solve()
        return self._sols

    def _eval_functional(self) -> Array:
        return self._adjoint_solver._functional(
            self._adjoint_solver._fwd_sol[:, None]
        )[None, :]

    def _jacobian(self, sample: Array) -> Array:
        self.set_param(sample[:, 0])
        if self._jacobian_mode == "backward":
            return self._jacobian_from_adjoint()
        return self._adjoint_solver.parameter_jacobian()

    def _jacobian_from_adjoint(self) -> Array:
        return self._adjoint_solver.gradient()[None, :]

    def _apply_hessian_from_adjoint(self, vec: Array) -> Array:
        return self._adjoint_solver.apply_hessian(vec)

    def __repr__(self):
        return "{0}(residual={1}, functional={2})".format(
            self.__class__.__name__,
            self._residual,
            self._functional,
        )


class SteadyAdjointModelFixedInitialIterate(SteadyAdjointModel):
    # Intended for testing only
    def __init__(
        self,
        residual: NewtonResidual,
        init_iterate: Array,
        nvars: int,
        functional: AdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        jacobian_implemented=False,
        apply_hessian_implemented=False,
        jacobian_mode: str = "backward",
    ):
        self._nvars = nvars
        super().__init__(
            residual, functional, newton_solver, jacobian_mode=jacobian_mode
        )
        self._jacobian_implemented = (
            jacobian_implemented or self._bkd.jacobian_implemented()
        )
        self._apply_hessian_implemented = (
            apply_hessian_implemented or self._bkd.hessian_implemented()
            )
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D Array")
        self._adjoint_solver.set_initial_iterate(init_iterate)

    def nvars(self) -> int:
        return self._nvars


class TransientAdjointModel(AdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        if not isinstance(time_residual, TimeIntegratorNewtonResidual):
            raise ValueError(
                "time_residual must be an instance of "
                "TimeIntegratorNewtonResidual"
            )

        self._time_residual = time_residual
        self._newton_solver = newton_solver
        self.setup_time_integrator(init_time, final_time, deltat)
        if functional is not None:
            self.set_functional(functional)

    def set_functional(self, functional: TransientAdjointFunctional):
        if not isinstance(functional, TransientAdjointFunctional):
            raise ValueError(
                "functional must be an instance of "
                "TransientAdjointFunctional"
            )
        self._functional = functional
        self._time_int.set_functional(self._functional)

    def setup_time_integrator(
        self, init_time: float, final_time: float, deltat: float
    ):
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._time_int = ImplicitTimeIntegrator(
            self._time_residual,
            self._init_time,
            self._final_time,
            self._deltat,
            newton_solver=self._newton_solver,
            verbosity=0,
        )

    @abstractmethod
    def get_initial_condition(self) -> Array:
        raise NotImplementedError

    def set_param(self, param: Array):
        if hasattr(self, "_functional"):
            # pass functional all parameters
            self._functional.set_param(param)
            # do not pass parameters unique to functional to native residual
            self._time_residual.native_residual.set_param(
                self._functional._residual_param(param)
            )
        else:
            self._time_residual.native_residual.set_param(param)

    def _eval_functional(self) -> Array:
        self._functional.set_quadrature_sample_weights(
            *self._time_residual.quadrature_samples_weights(self._times)
        )
        return self._functional(self._sols)[None, :]

    def _fwd_solve(self):
        init_sol = self.get_initial_condition()
        self._sols, self._times = self._time_int.solve(init_sol)

    def forward_solve(self, sample) -> Tuple[Array, Array]:
        self.set_param(sample[:, 0])
        self._fwd_solve()
        return self._sols, self._times

    def _jacobian_from_adjoint(self) -> Array:
        return self._time_int.gradient(self._sols, self._times)

    def __repr__(self):
        return "{0}(integrator={1})".format(
            self.__class__.__name__, self._time_int
        )
