from abc import ABC, abstractmethod
import textwrap

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.physics import Physics
from pyapprox.pde.collocation.functions import (
    MatrixOperator,
    TransientOperatorMixin,
)
from pyapprox.pde.collocation.newton import NewtonSolver, NewtonResidual
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    TimeIntegratorNewtonResidual,
    ImplicitTimeIntegrator,
)


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


class SteadyPhysicsNewtonResidual(NewtonResidual):
    def __init__(self, physics: Physics):
        super().__init__(physics._bkd)
        self._physics = physics

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _linsolve(self, sol_array: Array, res_array: Array) -> Array:
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    def __call__(self, sol_array: Array):
        res_array = self._physics._residual_array_from_solution_array(
            sol_array
        )
        return self._physics.apply_boundary_conditions_to_residual(
            sol_array, res_array
        )

    def jacobian(self, sol_array: Array):
        # assumes jac called after __call__
        jac = self._physics._residual_jacobian_from_solution_array(sol_array)
        return self._physics.apply_boundary_conditions_to_jacobian(
            sol_array, jac
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
        self.newton_solver.set_residual(
            SteadyPhysicsNewtonResidual(self.physics)
        )

    def solve(self, init_sol: MatrixOperator):
        init_sol_array = self._bkd.flatten(init_sol.get_values())
        sol_array = self.newton_solver.solve(init_sol_array)
        return self.physics.separate_solutions(sol_array)


class TransientPhysicsNewtonResidual(TransientNewtonResidual):
    def __init__(self, physics: Physics):
        super().__init__(physics._bkd)
        self._physics = physics

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _linsolve(self, sol_array: Array, res_array: Array) -> Array:
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    def __call__(self, sol_array: Array) -> Array:
        self._sol_array = sol_array
        return self._physics._residual_array_from_solution_array(sol_array)

    def jacobian(self, sol_array: Array) -> Array:
        return self._physics._residual_jacobian_from_solution_array(sol_array)

    def _apply_constraints_to_residual(self, res_array: Array) -> Array:
        return self._physics.apply_boundary_conditions_to_residual(
            self._sol_array, res_array
        )

    def _apply_constraints_to_jacobian(self, jac: Array) -> Array:
        return self._physics.apply_boundary_conditions_to_jacobian(
            self._sol_array, jac
        )

    def set_time(self, time: float):
        funs = self._physics.get_functions()
        for name, fun in funs.items():
            if isinstance(fun, TransientOperatorMixin):
                fun.set_time(time)

        for bndry in self._physics._bndrys:
            if isinstance(bndry._fun, TransientOperatorMixin):
                bndry._fun.set_time(time)


class TransientPDE(PDESolver):
    def __init__(self, physics: Physics, newton_solver: NewtonSolver = None):
        super().__init__(physics)
        if newton_solver is None:
            newton_solver = NewtonSolver()
        if not isinstance(newton_solver, NewtonSolver):
            raise ValueError(
                "newton_solver must be an instance of NewtonSolver"
            )

    def setup_time_integrator(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        init_time: float,
        final_time: float,
        deltat: float,
    ):
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        self._time_residual = time_residual_cls(
            TransientPhysicsNewtonResidual(self.physics)
        )
        self._time_residual._apply_constraints_to_residual = (
            self._time_residual.native_residual._apply_constraints_to_residual
        )
        self._time_residual._apply_constraints_to_jacobian = (
            self._time_residual.native_residual._apply_constraints_to_jacobian
        )
        self._time_int = ImplicitTimeIntegrator(
            self._time_residual,
            self._init_time,
            self._final_time,
            self._deltat,
            verbosity=0,
        )

    def solve(self, init_sol: MatrixOperator):
        self._sols, self._times = self._time_int.solve(init_sol.get_values())
        return self._sols, self._times
