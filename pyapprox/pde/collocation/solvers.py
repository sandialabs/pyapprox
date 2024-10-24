from abc import ABC, abstractmethod
import textwrap

from pyapprox.pde.collocation.physics import Physics
from pyapprox.pde.collocation.functions import MatrixFunction
from pyapprox.pde.collocation.newton import NewtonSolver


class PDESolver(ABC):
    def __init__(self, physics: Physics):
        if not isinstance(physics, Physics):
            raise ValueError("physics must be an instance of Physics")
        self._bkd = physics._bkd
        self.physics = physics

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}({1}\n)".format(
            self.__class__.__name__,
            textwrap.indent(
                f"\nphysics={self.physics},\nnetwon="
                + str(self.newton_solver),
                prefix="    ",
            ),
        )


class SteadyPDE(PDESolver):
    def __init__(self, physics: Physics, newton_solver: NewtonSolver = None):
        super().__init__(physics)
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self.newton_solver = newton_solver
        self.newton_solver.set_residual(self.physics)
    
    def solve(self, init_sol: MatrixFunction):
        init_sol_array = self._bkd.flatten(init_sol.get_values())
        sol_array = self.newton_solver.solve(init_sol_array)
        return self.physics.separate_solutions(sol_array)


# class TransientPhysicsNewtonResidual(TransientNewtonResidual):
#     def __init__(self, physics: Physics):
#         super().__init__(physics._bkd)
#         self._physics = physics

#     def linsolve(self, iterate, res):
#         self.physics.linsolve(iterate, res)

#     def __call__(self, iterate):
#         raise NotImplementedError
    
    
# class TransientPDE(PDESolver):
#     def __init__(
#             self,
#             physics: Physics,
#             time_integrator: TimeIntegrator,
#     ):
#         super().__init__(physics, newton_solver)
#         if not isinstance(time_integrator, TimeIntegrator):
#             raise ValueError(
#                 "time_integrator must be an instance of TimeIntegrator"
#             )
#         self._time_integrator = time_integrator

#     def solve(self, init_sol: MatrixFunction):
#         sols, times = [], []
#         time = init_time
#         times.append(time)
#         sol = init_sol.copy()
#         sols.append(init_sol[:, None])
#         while time < final_time-1e-12:
#             if verbosity >= 1:
#                 print("Time", time)
#             deltat = min(self._deltat, final_time-time)
#             sol = self._update(
#                 sol, time, deltat, sol.copy())
#             sols.append(sol[:, None])
#             time += deltat
#             times.append(time)
#         if verbosity >= 1:
#             print("Time", time)
#         sols = np.hstack(sols)
