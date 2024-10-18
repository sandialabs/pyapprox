from abc import ABC, abstractmethod
import textwrap

from pyapprox.pde.collocation.physics import Physics
from pyapprox.pde.collocation.functions import MatrixFunction
from pyapprox.pde.collocation.newton import NewtonSolver


class PDESolver(ABC):
    def __init__(self, physics: Physics, newton_solver: NewtonSolver = None):
        if not isinstance(physics, Physics):
            raise ValueError("physics must be an instance of Physics")
        self._bkd = physics._bkd
        self.physics = physics
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )
        self.newton_solver = newton_solver
        self.newton_solver.set_residual(self.physics)

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}({1}\n)".format(
            self.__class__.__name__,
            textwrap.indent(
                f"\nphysics={self.physics},\nnetwon="+str(self.newton_solver),
                prefix="    ")
        )


class SteadyStatePDE(PDESolver):
    def solve(self, init_sol: MatrixFunction):
        init_sol_array = self._bkd.flatten(init_sol.get_values())
        sol_array = self.newton_solver.solve(init_sol_array)
        return self.physics.separate_solutions(sol_array)
