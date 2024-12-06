from abc import ABC, abstractmethod
import textwrap
from typing import Tuple, Union

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.physics import Physics
from pyapprox.pde.collocation.functions import (
    ScalarOperator,
    MatrixOperator,
    TransientOperatorMixin,
    OrthogonalCoordinateCollocationBasis,
)
from pyapprox.pde.collocation.newton import NewtonSolver, NewtonResidual
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    TimeIntegratorNewtonResidual,
    BackwardEulerResidual,
)
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointFunctional, TransientAdjointModel, SteadyAdjointModel
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


class SteadyForwardCollocationModelFromPhysics(SteadyAdjointModel):
    # Only intended for testing with manufactured solutions
    def __init__(
        self,
        physics: Physics,
        newton_solver: NewtonSolver = None,
    ):
        self._physics = physics
        super().__init__(
            SteadyPhysicsNewtonResidual(physics), newton_solver=newton_solver
        )

    def nvars(self) -> int:
        return 0

    def forward_solve(self, init_cond: Union[MatrixOperator, ScalarOperator]):
        sol_array = self._adjoint_solver._newton_solver.solve(
            init_cond.get_flattened_values()
        )
        return self._physics.solution_from_array(sol_array)


class TransientPhysicsNewtonResidual(TransientNewtonResidual):
    def __init__(self, physics: Physics):
        super().__init__(physics._bkd)
        self._physics = physics

    def mass_matrix(self, nterms: int) -> Array:
        return self._physics.mass_matrix(nterms)

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
            # TODO CREATE A BASE CLASS WHICH DERIVES CONSTANT AND FUNCTION based bndry
            if hasattr(bndry, "_fun") and isinstance(
                    bndry._fun, TransientOperatorMixin
            ):
                bndry._fun.set_time(time)


class TransientAdjointCollocationModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        newton_solver: NewtonSolver = None,
        functional: TransientAdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self.setup_basis()
        self.setup_physics()
        self.setup_boundaries()

        time_residual = time_residual_cls(
            TransientPhysicsNewtonResidual(self._physics)
        )
        time_residual._apply_constraints_to_residual = (
            time_residual.native_residual._apply_constraints_to_residual
        )
        time_residual._apply_constraints_to_jacobian = (
            time_residual.native_residual._apply_constraints_to_jacobian
        )

        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual,
            functional,
            newton_solver,
            backend=backend,
        )

    @abstractmethod
    def setup_basis(self):
        raise NotImplementedError

    @abstractmethod
    def setup_physics(self):
        raise NotImplementedError

    @abstractmethod
    def setup_boundaries(self):
        raise NotImplementedError

    def basis(self) -> OrthogonalCoordinateCollocationBasis:
        return self._basis

    def physics(self) -> Physics:
        return self._physics


class TransientForwardCollocationModelFromPhysics(
        TransientAdjointCollocationModel
):
    # Only intended for testing with manufactured solutions
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        physics: Physics,
        newton_solver: NewtonSolver = None,
    ):
        self._physics = physics
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls,
            newton_solver,
            functional=None,
            backend=self._physics._bkd
        )

    def nvars(self) -> int:
        return 0

    def get_initial_condition(self) -> Array:
        raise NotImplementedError(
            "initial condition must be passed to forward_solve")

    def setup_physics(self):
        if not isinstance(self._physics, Physics):
            raise ValueError("physics must be and instance of Physics")

    def setup_boundaries(self):
        if not hasattr(self._physics, "_bndrys"):
            raise ValueError("physics boundaries were not set")

    def setup_basis(self):
        self._basis = self.physics().basis()

    def forward_solve(self, init_cond: Union[MatrixOperator, ScalarOperator]):
        sol, times = self._time_int.solve(init_cond.get_flattened_values())
        values_shape = (
            self.physics().ncomponents(),
            self.basis().mesh().nmesh_pts(),
            times.shape[0]
        )
        sol = self._bkd.reshape(sol, values_shape)
        return sol, times


class SplitPhysicsTimeIntegratorNewtonResidual(
        TimeIntegratorNewtonResidual
):
    def __init__(
        self,
        transient_residual: TransientPhysicsNewtonResidual,
        # TODO: set backward euler as default only to enable easier testing,
        # when this class works remove default
        time_residual_cls: TimeIntegratorNewtonResidual = BackwardEulerResidual,
    ):
        super().__init__(transient_residual)
        # SplitPhysicsMixin is structured so that while passing in all physics
        # time residual is only applied to the traansient physics
        self._time_residual = time_residual_cls(transient_residual)
        self._physics = self._time_residual.native_residual._physics

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        return self._time_residual.quadrature_samples_weights(times)

    def __call__(self, sol_array: Array) -> Array:
        # called by TimeIntegratorNewtonResidual.__call__ which is called by
        # ImplicitTimeIntegrator.step()
        split_physics = self._time_residual.native_residual._physics
        split_physics._set_steady_and_transient_components(sol_array)
        transient_sol_array = split_physics._transient_physics_solution_from_array(
            sol_array
        ).get_flattened_values()
        # self._time_residual._value(sol) calls native_residual.__call__ which
        # calls _residual_array_from_solution_array
        res = self._bkd.hstack(
            (self._time_residual._value(transient_sol_array),
             self._physics.steady_value(sol_array))
        )
        # must overwrite self._time_residual.native_residual._sol_array
        # which was set to just transient physics above
        self._time_residual.native_residual._sol_array = sol_array
        res = self._apply_constraints_to_residual(res)
        return res

    def _jacobian(self, sol_array: Array) -> Array:
        split_physics = self._time_residual.native_residual._physics
        split_physics._set_steady_and_transient_components(sol_array)
        transient_sol_array = split_physics._transient_physics_solution_from_array(
            sol_array
        ).get_flattened_values()
        steady_jac = self._physics.steady_jacobian(sol_array)
        transient_jac = self._time_residual._jacobian(transient_sol_array)
        jac = self._bkd.vstack((transient_jac, steady_jac))
        # must overwrite self._time_residual.native_residual._sol_array
        # which was set to just transient physics above
        self._time_residual.native_residual._sol_array = sol_array
        jac = self._apply_constraints_to_jacobian(jac)
        return jac

    def set_time(self, time: float, deltat: float, prev_sol: Array):
        self._time = time
        self._deltat = deltat
        # self._prev_sol = prev_sol
        split_physics = self._time_residual.native_residual._physics
        self._prev_sol = split_physics._transient_physics_solution_from_array(
            prev_sol
        ).get_flattened_values()
        self._time_residual.set_time(time, deltat, self._prev_sol)

    def _apply_constraints_to_residual(self, res_array: Array) -> Array:
        # boundary conditions applied by each physics component
        raise NotImplementedError

    def _apply_constraints_to_jacobian(self, jac: Array) -> Array:
        # boundary conditions applied by each physics component
        raise NotImplementedError
