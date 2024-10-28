from abc import ABC, abstractmethod
from typing import Tuple

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.newton import NewtonSolver, NewtonResidual


class TransientNewtonResidual(NewtonResidual):
    @abstractmethod
    def set_time(self, time: float):
        raise NotImplementedError

    def linsolve(self, sol: Array, res: Array):
        return self._bkd.solve(self.jacobian(sol), res)

    def mass_matrix(self, nstates):
        return self._bkd.eye(nstates)

    def _initial_residual_param_jacobian(self):
        raise NotImplementedError


class TimeIntegratorNewtonResidual(NewtonResidual):
    def __init__(self, residual: NewtonResidual):
        super().__init__(residual._bkd)
        self._residual = residual

    def set_time(self, time: float, deltat: float, prev_sol: Array):
        self._time = time
        self._residual.set_time(self._time)
        self._deltat = deltat
        self._prev_sol = prev_sol

    def linsolve(self, sol: Array, res: Array):
        return self._bkd.solve(self.jacobian(sol), res)

    @abstractmethod
    def _residual_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def adjoint_initial_condition(self, fwd_sol: Array, dqdu: Array) -> Array:
        drdu = self.jacobian(fwd_sol)
        return self._bkd.solve(drdu.T, -dqdu)

    def adjoint_final_solution(
            self, fwd_sol: Array, prev_adj_sol: Array, dqdu: Array
    ) -> Array:
        return prev_adj_sol

    def initial_residual_param_jacobian(self) -> Array:
        jac = self._residual._initial_residual_param_jacobian()
        if jac.ndim != 2:
            raise RuntimeError(f"jac has the wrong shape {jac.shape}")
        return jac


class ExplicitTimeIntegratorNewtonResidual(TimeIntegratorNewtonResidual):
    def jacobian(self, sol: Array) -> Array:
        # todo: do not solve linear system when using explicit time integrators
        return self._residual.mass_matrix(sol.shape[0])

    def adjoint_initial_condition(self, fwd_sol: Array, dqdu: Array) -> Array:
        return -dqdu

    def adjoint_final_solution(
            self, fwd_sol: Array, prev_adj_sol: Array, dqdu: Array
    ) -> Array:
        # prev_adjoint_sol is the adj_sol computed last.
        # because we are evolving backward in time prev_adj_sol is at a larger time
        # than init_time set here
        return self.jacobian(fwd_sol) @ prev_adj_sol - dqdu


class ForwardEulerResidual(ExplicitTimeIntegratorNewtonResidual):
    # Left sided piecewise constant integration
    def __call__(self, sol: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return (
            sol
            - self._prev_sol
            - self._deltat * self._residual(self._prev_sol)
        )

    def adjoint_residual(self, sol: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return sol + self._prev_sol - self._deltat * self._residual(sol)

    def _residual_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError


class BackwardEulerResidual(TimeIntegratorNewtonResidual):
    # Right sided piecewise constant integration
    def __call__(self, sol: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return sol - self._prev_sol - self._deltat * self._residual(sol)

    def jacobian(self, sol: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return self._residual.mass_matrix(
            sol.shape[0]
        ) - self._deltat * self._residual.jacobian(sol)

    def adjoint_residual(self, sol: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return sol + self._prev_sol - self._deltat * self._residual(sol)

    def _residual_param_jacobian(self, sol: Array) -> Array:
        return -self._deltat * self._residual.residual_param_jacobian(sol)


class HeunResidual(ExplicitTimeIntegratorNewtonResidual):
    # Trapezoid integration
    def __call__(self, sol: Array):
        self._residual.set_time(self._time)
        current_res = self._residual(self._prev_sol)
        next_sol = self._prev_sol + self._deltat * current_res
        self._residual.set_time(self._time + self._deltat)
        next_res = self._residual(next_sol)
        return (
            sol
            - self._prev_sol
            - 0.5 * self._deltat * (current_res + next_res)
        )

    def _residual_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError


class CrankNicholsonResidual(TimeIntegratorNewtonResidual):
    # Trapezoid integration
    def __call__(self, sol: Array):
        self._residual.set_time(self._time)
        current_res = self._residual(self._prev_sol)
        self._residual.set_time(self._time + self._deltat)
        next_res = self._residual(sol)
        return (
            sol
            - self._prev_sol
            - 0.5 * self._deltat * (current_res + next_res)
        )

    def jacobian(self, sol: Array):
        self._residual.set_time(self._time)
        current_jac = self._residual.jacobian(self._prev_sol)
        self._residual.set_time(self._time + self._deltat)
        next_jac = self._residual.jacobian(sol)
        return self._residual.mass_matrix(
            sol.shape[0]
        ) - 0.5 * self._deltat * (current_jac + next_jac)

    def _residual_param_jacobian(self, sol: Array) -> Array:
        self._residual.set_time(self._time)
        current_param_jac = self._residual.residual_param_jacobian(sol)
        self._residual.set_time(self._time + self._deltat)
        next_param_jac = self._residual.residual_param_jacobian(sol)
        return -0.5 * self._deltat * (current_param_jac + next_param_jac)


class RK4(ExplicitTimeIntegratorNewtonResidual):
    # Simpsons integration
    def __call__(self, sol: Array):
        self._residual.set_time(self._time)
        k1_res = self._residual(self._prev_sol)
        self._residual.set_time(self._time + self._deltat / 2)
        k2_sol = self._prev_sol + self._deltat * k1_res / 2
        k2_res = self._residual(k2_sol)
        k3_sol = self._prev_sol + self._deltat * k2_res / 2
        k3_res = self._residual(k3_sol)
        self._residual.set_time(self._time + self._deltat)
        k4_sol = self._prev_sol + self._deltat * k3_res
        k4_res = self._residual(k4_sol)
        return (
            sol
            - self._prev_sol
            - self._deltat / 6 * (k1_res + 2 * k2_res + 2 * k3_res + k4_res)
        )

    def _residual_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError


class Functional(ABC):
    def __init__(self, backend=NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def nstates(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def nparams(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _value(self, sol: Array) -> Array:
        raise NotImplementedError

    def __call__(self, sol: Array) -> Array:
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        val = self._value(sol)
        if val.ndim != 1:
            raise RuntimeError(f"{self} must return a 1D array")
        return val

    def __repr__(self):
        return "{0}(nstates={1}, nparams={2})".format(
            self.__class__.__name__, self.nstates(), self.nparams()
        )


class AdjointFunctional(Functional):
    @abstractmethod
    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def qoi_sol_jacobian(self, sol: Array) -> Array:
        """Gradient of qoi with respect to solution"""
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        jac = self._qoi_sol_jacobian(sol)
        if jac.shape[1] != sol.shape[1]:
            raise RuntimeError("jac has the wrong shape")
        return jac

    @abstractmethod
    def _qoi_param_jacobian(self, sol: Array) -> Array:
        raise NotImplementedError

    def qoi_param_jacobian(self, sol: Array) -> Array:
        """Gradient of QoI with respect to parameters"""
        if sol.ndim != 2 or sol.shape[0] != self.nstates():
            raise ValueError("sol must be a 2d Array")
        jac = self._qoi_param_jacobian(sol)
        if jac.ndim != 1 or jac.shape[0] != self.nparams():
            raise RuntimeError("jac has the wrong shape")
        return jac

    def set_param(self, param: Array):
        if param.ndim != 1:
            raise ValueError("param must be a 1D Array")
        self._param = param


class ImplicitTimeIntegrator:
    def __init__(
        self,
        residual: TimeIntegratorNewtonResidual,
        init_time: float,
        final_time: float,
        deltat: float,
        newton_solver: NewtonSolver = None,
        verbosity: int = 0,
    ):
        if not isinstance(residual, TimeIntegratorNewtonResidual):
            raise ValueError(
                "residual must be an instance of TimeIntegratorNewtonResidual"
            )
        self._bkd = residual._bkd
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._verbosity = verbosity
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self.newton_solver = newton_solver
        self.newton_solver.set_residual(residual)

    def set_functional(self, functional: AdjointFunctional):
        if not isinstance(functional, AdjointFunctional):
            raise ValueError(
                "functional must be an instance of AdjointFunctional"
            )
        self._functional = functional

    def step(self, sol: Array, deltat: float) -> Array:
        self.newton_solver._residual.set_time(self._time, deltat, sol)
        sol = self.newton_solver.solve(self._bkd.copy(sol))
        self._time += deltat
        if self._verbosity >= 1:
            print("Time", self._time)
        return sol

    def solve(self, init_sol: Array) -> Tuple[Array, Array]:
        sols, times = [], []
        self._time = self._init_time
        times.append(self._time)
        sol = self._bkd.copy(init_sol)
        sols.append(init_sol)
        while self._time < self._final_time - 1e-12:
            deltat = min(self._deltat, self._final_time - self._time)
            sol = self.step(sol, deltat)
            sols.append(sol)
            times.append(self._time)
        sols = self._bkd.stack(sols, axis=1)
        return sols, self._bkd.array(times)

    def adjoint_step(
        self,
        fwd_sol: Array,
        reverse_deltat: float,
        dqdu: Array,
        adj_sol: Array,
    ) -> Array:
        self.newton_solver._residual.set_time(
            self._time, reverse_deltat, fwd_sol
        )
        # jacobian of residual with respect to solution
        drdu = self.newton_solver._residual.jacobian(fwd_sol)
        # TODO below will need to allow for assembly with FEM models
        adj_sol = self._bkd.solve(drdu.T, adj_sol-dqdu)
        return adj_sol

    def solve_adjoint(self, fwd_sols: Array, times: Array) -> Array:
        if not self._bkd.allclose(
            times[-1], self._bkd.atleast1d(self._final_time), atol=1e-12
        ):
            raise ValueError("times array is inconsistent with final_time")
        # copy required when using torch
        self._time = self._bkd.copy(times)[-1]
        # todo compute dqdu at each time step rather than all upfront
        dqdu = self._functional.qoi_sol_jacobian(fwd_sols)
        adj_sols = self._bkd.empty(fwd_sols.shape)
        deltat = times[-1] - times[-2]
        self.newton_solver._residual.set_time(
            self._time, deltat, fwd_sols[:, -1]
        )
        adj_sols[:, -1] = (
            self.newton_solver._residual.adjoint_initial_condition(
                fwd_sols[:, -1], dqdu[:, -1]
            )
        )
        for ii in range(fwd_sols.shape[1] - 2, 0, -1):
            deltat = times[ii] - times[ii - 1]
            adj_sols[:, ii] = self.adjoint_step(
                fwd_sols[:, ii + 1],
                deltat,
                dqdu[:, ii],
                adj_sols[:, ii + 1],
            )
            self._time -= self._deltat
        adj_sols[:, 0] = self.newton_solver._residual.adjoint_final_solution(
            fwd_sols[:, 0], adj_sols[:, 1], dqdu[:, 0]
        )
        self._time -= self._deltat
        return adj_sols

    def gradient(self, fwd_sols: Array, times: Array) -> Array:
        adj_sols = self.solve_adjoint(fwd_sols, times)
        dqdp = self._functional.qoi_param_jacobian(fwd_sols)
        grad = dqdp
        residual = self.newton_solver._residual
        # fwd_sols[:, 0] will never be used
        residual.set_time(
            times[0], times[1]-times[0], fwd_sols[:, 0]
        )
        drdp = residual.initial_residual_param_jacobian()
        grad += adj_sols[:, 0] @ drdp
        for ii, time in enumerate(times[1:], start=1):
            residual.set_time(
                times[ii], times[ii]-times[ii-1], fwd_sols[:, ii-1]
            )
            drdp = residual.residual_param_jacobian(
                fwd_sols[:, ii]
            )
            grad += adj_sols[:, ii] @ drdp
        return self._bkd.atleast2d(grad)


from pyapprox.interface.model import SingleSampleModel


class AdjointModel(SingleSampleModel):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._jacobian_implemented = True
        self._setup_residual()
        self._setup_functional()

    def nqoi(self):
        return 1

    @abstractmethod
    def _fwd_solve(self):
        raise NotImplementedError

    @abstractmethod
    def _gradient(self):
        raise NotImplementedError

    @abstractmethod
    def _setup_residual(self):
        raise NotImplementedError

    @abstractmethod
    def _setup_functional(self):
        raise NotImplementedError

    @abstractmethod
    def set_param(self):
        raise NotImplementedError

    def _evaluate(self, sample: Array) -> Array:
        self._sample = sample
        self.set_param(sample)
        self._fwd_solve()
        return self.functional(self._sols)[None, :]

    def _jacobian(self, sample: Array):
        if not self._bkd.allclose(sample, self._sample, atol=1e-15):
            self._evaluate(sample)
        return self._time_int.gradient(self._sols, self._times)


class TransientAdjointModel(AdjointModel):
    def __init__(self, backend=NumpyLinAlgMixin):
        super().__init__(backend)
        self._setup_time_integrator()
        self._time_int.set_functional(self.functional)

    @abstractmethod
    def _setup_time_integrator(self):
        raise NotImplementedError

    def _fwd_solve(self):
        init_sol = self.get_initial_solution()
        self._sols, self._times = self._time_int.solve(init_sol)

    def _gradient(self):
        return self._time_int.gradient(self._sols, self._times)
