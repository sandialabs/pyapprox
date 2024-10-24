from abc import ABC, abstractmethod

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.newton import NewtonSolver, NewtonResidual


class TransientNewtonResidual(NewtonResidual):
    @abstractmethod
    def set_time(self, time: float):
        raise NotImplementedError

    def linsolve(self, sol: Array, res: Array):
        return self._bkd.solve(self.jacobian(sol), res)

    def mass_matrix(self, nstates):
        return self._bkd.eye(nstates)


class TimeIntegratorNewtonResidual(NewtonResidual):
    def __init__(self, residual: NewtonResidual):
        super().__init__(residual._bkd)
        self._residual = residual

    def set_time(self, time: float, deltat: float, prev_sol: Array):
        self._time = time
        self._deltat = deltat
        self._prev_sol = prev_sol

    def linsolve(self, sol: Array, res: Array):
        return self._bkd.solve(self.jacobian(sol), res)


class BackwardEulerResidual(TimeIntegratorNewtonResidual):
    def __call__(self, sol: Array):
        self._residual.set_time(self._time+self._deltat)
        return sol - self._prev_sol - self._deltat * self._residual(sol)

    def jacobian(self, sol: Array):
        self._residual.set_time(self._time+self._deltat)
        return (
            self._residual.mass_matrix(sol.shape[0])
            - self._deltat * self._residual.jacobian(sol)
        )


class CrankNicholsonResidual(TimeIntegratorNewtonResidual):
    def __call__(self, sol: Array):
        self._residual.set_time(self._time)
        current_res = self._residual(self._prev_sol)
        self._residual.set_time(self._time+self._deltat)
        next_res = self._residual(sol)
        return sol - self._prev_sol - 0.5*self._deltat * (
            current_res + next_res
        )

    def jacobian(self, sol: Array):
        self._residual.set_time(self._time)
        current_jac = self._residual.jacobian(self._prev_sol)
        self._residual.set_time(self._time+self._deltat)
        next_jac = self._residual.jacobian(sol)
        return (
            self._residual.mass_matrix(sol.shape[0])
            - 0.5*self._deltat * (current_jac + next_jac)
        )


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

    def step(self, sol: Array, deltat: float) -> Array:
        self.newton_solver._residual.set_time(
            self._time, deltat, sol
        )
        sol = self.newton_solver.solve(self._bkd.copy(sol))
        self._time += deltat
        if self._verbosity >= 1:
            print("Time", self._time)
        return sol

    def solve(self, init_sol: Array):
        sols, times = [], []
        self._time = self._init_time
        times.append(self._time)
        sol = self._bkd.copy(init_sol)
        sols.append(init_sol)
        while self._time < self._final_time-1e-12:
            deltat = min(self._deltat, self._final_time-self._time)
            sol = self.step(sol, deltat)
            sols.append(sol)
            times.append(self._time)
        sols = self._bkd.stack(sols, axis=1)
        return sols, self._bkd.array(times)
