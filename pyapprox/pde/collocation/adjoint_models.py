from abc import abstractmethod

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


class AdjointModel(SingleSampleModel):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True

    def nqoi(self):
        return 1

    @abstractmethod
    def _fwd_solve(self):
        raise NotImplementedError

    @abstractmethod
    def _set_param(self, param: Array):
        raise NotImplementedError

    @abstractmethod
    def _eval_functional(self):
        raise NotImplementedError

    def _evaluate(self, sample: Array) -> Array:
        self._set_param(sample[:, 0])
        self._fwd_solve()
        return self._eval_functional()

    @abstractmethod
    def _jacobian_from_adjoint(self):
        raise NotImplementedError

    def _jacobian(self, sample: Array):
        self._set_param(sample[:, 0])
        return self._jacobian_from_adjoint()

    def _apply_hessian_from_adjoint(self, vec: Array):
        raise NotImplementedError("_hessian_from_adjoint not implemented")

    def _apply_hessian(self, sample: Array, vec: Array):
        self._set_param(sample[:, 0])
        return self._apply_hessian_from_adjoint(vec[:, 0])


class SteadyAdjointModel(AdjointModel):
    def __init__(
        self,
        residual: NewtonResidual,
        functional: AdjointFunctional,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
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

    @abstractmethod
    def get_initial_iterate(self) -> Array:
        raise NotImplementedError

    def _set_param(self, param: Array):
        self._adjoint_solver.set_param(param)

    def _fwd_solve(self):
        self._adjoint_solver.forward_solve()

    def _eval_functional(self):
        return self._adjoint_solver._functional(
            self._adjoint_solver._fwd_sol[:, None]
        )[None, :]

    def _jacobian_from_adjoint(self):
        return self._adjoint_solver.gradient()[None, :]

    def _apply_hessian_from_adjoint(self, vec: Array):
        return self._adjoint_solver.apply_hessian(vec)

    def __repr__(self):
        return "{0}(residual={1}, functional={2})".format(
            self.__class__.__name__,
            self._residual,
            self._functional,
        )


class SteadyAdjointModelFixedInitialIterate(SteadyAdjointModel):
    def __init__(
        self,
        residual: NewtonResidual,
        functional: AdjointFunctional,
        init_iterate: Array,
        newton_solver: NewtonSolver = None,
        apply_hessian_implemented=False,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(residual, functional, newton_solver, backend)
        self._apply_hessian_implemented = apply_hessian_implemented
        if init_iterate.ndim != 1:
            raise ValueError("init_iterate must be 1D Array")
        self._adjoint_solver.set_initial_iterate(init_iterate)

    def get_initial_iterate(self) -> Array:
        return self._init_iterate


class TransientAdjointModel(AdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(backend)
        if not isinstance(time_residual, TimeIntegratorNewtonResidual):
            raise ValueError(
                "time_residual must be an instance of "
                "TimeIntegratorNewtonResidual"
            )
        if not isinstance(functional, TransientAdjointFunctional):
            raise ValueError(
                "functional must be an instance of "
                "TransientAdjointFunctional"
            )
        self._time_residual = time_residual
        self._functional = functional
        self.setup_time_integrator(init_time, final_time, deltat)

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
            verbosity=0,
        )
        self._time_int.set_functional(self._functional)

    @abstractmethod
    def get_initial_condition(self) -> Array:
        raise NotImplementedError

    def _set_param(self, param: Array):
        self._time_residual.native_residual.set_param(param)
        self._functional.set_param(param)

    def _eval_functional(self):
        self._functional.set_quadrature_sample_weights(
            *self._time_residual.quadrature_samples_weights(self._times)
        )
        return self._functional(self._sols)[None, :]

    def _fwd_solve(self):
        init_sol = self.get_initial_condition()
        self._sols, self._times = self._time_int.solve(init_sol)

    def _jacobian_from_adjoint(self):
        return self._time_int.gradient(self._sols, self._times)

    def __repr__(self):
        return "{0}(integrator={1})".format(
            self.__class__.__name__, self._time_int
        )
