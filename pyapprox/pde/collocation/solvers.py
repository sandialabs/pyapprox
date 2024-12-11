from abc import ABC, abstractmethod
import textwrap
from typing import Tuple, Union

from pyapprox.util.linearalgebra.linalgbase import Array, LinAlgMixin
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.physics import (
    Physics,
    SteadyPhysicsNewtonResidualMixin,
    TransientPhysicsNewtonResidualMixin,
    TransientSplitPhysicsNewtonResidualMixin,
)
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


class SteadyForwardCollocationModelFromPhysics(SteadyAdjointModel):
    # Only intended for testing with manufactured solutions
    def __init__(
        self,
        physics: Physics,
        newton_solver: NewtonSolver = None,
    ):
        if not isinstance(physics, SteadyPhysicsNewtonResidualMixin):
            raise ValueError(
                "physics must be an instance of "
                "SteadyPhysicsNewtonResidualMixin"
            )
        self._physics = physics
        super().__init__(
            physics, newton_solver=newton_solver
        )

    def nvars(self) -> int:
        return 0

    def forward_solve(self, init_cond: Union[MatrixOperator, ScalarOperator]):
        sol_array = self._adjoint_solver._newton_solver.solve(
            init_cond.get_flattened_values()
        )
        return self._physics.solution_from_array(sol_array)


class CollocationModelMixin(ABC):
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

    def nvars(self) -> int:
        if not hasattr(self, "_functional") or self._functional is None:
            return self.physics().nvars()
        return (
            self._functional.nunique_functional_params()
            + self.physics().nvars()
        )


class SteadyAdjointCollocationModel(CollocationModelMixin, SteadyAdjointModel):
    def __init__(
        self,
        newton_solver: NewtonSolver = None,
        functional: TransientAdjointFunctional = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._bkd = backend
        self.setup_basis()
        self.setup_physics()
        self.setup_boundaries()

        if not isinstance(self._physics, SteadyPhysicsNewtonResidualMixin):
            raise ValueError(
                "physics must be an instance of "
                "SteadyPhysicsNewtonResidualMixin"
            )

        super().__init__(
            self._physics,
            functional,
            newton_solver
        )

    def forward_solve(self, sample) -> Array:
        super().forward_solve(sample)
        return self.physics().solution_from_array(self._sols)


class TransientAdjointCollocationModel(
        TransientAdjointModel, CollocationModelMixin
):
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

        if not isinstance(self._physics, TransientPhysicsNewtonResidualMixin):
            raise ValueError(
                "physics must be an instance of "
                "TransientPhysicsNewtonResidualMixin"
            )

        time_residual = time_residual_cls(
            self._physics
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
        if not isinstance(physics, TransientPhysicsNewtonResidualMixin):
            raise ValueError(
                "physics must be an instance of "
                "TransientPhysicsNewtonResidualMixin"
            )

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
        transient_residual: TransientSplitPhysicsNewtonResidualMixin,
        # TODO: set backward euler as default only to enable easier testing,
        # when this class works remove default
        time_residual_cls: TimeIntegratorNewtonResidual = BackwardEulerResidual,
    ):
        if not isinstance(
                transient_residual, TransientSplitPhysicsNewtonResidualMixin
        ):
            raise ValueError(
                "transient_residual must be and instance of "
                "TransientSplitPhysicsNewtonResidualMixin"
            )

        super().__init__(transient_residual)
        # SplitPhysicsMixin is structured so that while passing in all physics
        # time residual is only applied to the transient physics
        self._time_residual = time_residual_cls(transient_residual)
        self._physics = self._time_residual.native_residual

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        return self._time_residual.quadrature_samples_weights(times)

    def __call__(self, sol_array: Array) -> Array:
        # called by TimeIntegratorNewtonResidual.__call__ which is called by
        # ImplicitTimeIntegrator.step()
        split_physics = self._time_residual.native_residual
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
        split_physics = self._time_residual.native_residual
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
        split_physics = self._time_residual.native_residual
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
