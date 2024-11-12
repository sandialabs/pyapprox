from abc import abstractmethod
from typing import Dict, List

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.functions import (
    MatrixOperator,
    ScalarSolution,
    VectorSolution,
    ScalarFunction,
    ScalarOperator,
    VectorFunction,
    SolutionMixin,
    Operator,
    ScalarFunctionFromCallable,
    ScalarMonomialOperator,
)
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.pde.collocation.boundaryconditions import (
    BoundaryOperator,
    PeriodicBoundary,
    RobinBoundary,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis
from pyapprox.pde.collocation.functions import nabla, div, sqmagnitude


class Physics(NewtonResidual):
    def __init__(self, basis: OrthogonalCoordinateCollocationBasis):
        super().__init__(basis._bkd)
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self.basis = basis
        self._flux_jacobian_implemented = False

    def mass_matrix(self, nterms: int) -> Array:
        return self._bkd.eye(nterms)

    def set_boundaries(self, bndrys: list[BoundaryOperator]):
        nperiodic_boundaries = 0
        for bndry in bndrys:
            if isinstance(bndry, PeriodicBoundary):
                nperiodic_boundaries += 1
        if len(bndrys) + nperiodic_boundaries != len(self.basis.mesh._bndrys):
            raise ValueError("Must set all boundaries")
        self._bndrys = bndrys
        for bndry in self._bndrys:
            if isinstance(bndry, RobinBoundary):
                if not self._flux_jacobian_implemented:
                    raise ValueError(
                        f"RobinBoundary requested but {self} "
                        "does not define _flux_jacobian"
                    )
                bndry.set_flux_functions(
                    self._flux_from_array, self._flux_jacobian_from_array
                )

    def apply_boundary_conditions_to_residual(
        self, sol_array: Array, res_array: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_residual(sol_array, res_array)
        return res_array

    def apply_boundary_conditions_to_jacobian(
        self, sol_array: Array, jac: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_jacobian(sol_array, jac)
        return res_array

    @abstractmethod
    def residual(self, sol: MatrixOperator):
        raise NotImplementedError

    def _residual_function_from_solution_array(self, sol_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        sol = self._separate_solutions(sol_array)
        return self.residual(sol)

    def _residual_array_from_solution_array(self, sol_array: Array):
        # TODO add option to matrix function that stops jacobian being computed
        # when computing residual. useful for explicit time stepping
        # and linear problems where jacobian does not depend on time or
        # on uncertain parameters of PDE
        res = self._residual_function_from_solution_array(sol_array)
        res_array = self._bkd.flatten(res.get_values())
        return res_array

    def _residual_jacobian_from_solution_array(self, sol_array: Array):
        res = self._residual_function_from_solution_array(sol_array)
        jac = res.get_matrix_jacobian()
        return self._bkd.reshape(
            jac, (jac.shape[0] * jac.shape[2], jac.shape[3])
        )

    @abstractmethod
    def _separate_solutions(self, array):
        raise NotImplementedError

    def separate_solutions(self, sol_array: Array):
        sol = self._separate_solutions(sol_array)
        if not isinstance(sol, SolutionMixin) or not isinstance(
            sol, MatrixOperator
        ):
            raise RuntimeError(
                "sol must be derived from SolutionMixin and MatrixOperator"
            )
        return sol

    # TODO consider making physics not derive from newton residual.
    # As we already wrap physics in another class that acts as the newton residual
    def __call__(self, sol_array: Array):
        raise NotImplementedError(
            "Use physics with SteadyPhysicsNewtonResidual"
        )

    def jacobian(self, sol_array: Array):
        raise NotImplementedError(
            "Use physics with SteadyPhysicsNewtonResidual"
        )

    def _flux(self, sol_array: Array):
        raise NotImplementedError

    def _flux_jacobian_from_array(self, sol_array: Array):
        sol = self.separate_solutions(sol_array)
        flux_jac = self._flux(sol).get_matrix_jacobian()
        return flux_jac[:, 0, :, :]

    def _flux_from_array(self, sol_array: Array):
        sol = self._separate_solutions(sol_array)
        flux = self._flux(sol).get_matrix_values()
        if flux.shape[1] != 1:
            raise RuntimeError("flux must be a vector valued funciton")
        return flux[:, 0, :]

    def get_functions(self) -> Dict[str, MatrixOperator]:
        funs = dict()
        return funs

    def __repr__(self):
        return "{0}(bndrys={1})".format(self.__class__.__name__, self._bndrys)


class ScalarPhysicsMixin:
    def _separate_solutions(self, array: Array):
        sol = ScalarSolution(self.basis)
        sol.set_values(array)
        sol.set_matrix_jacobian(sol._initial_matrix_jacobian())
        return sol

    def _check_is_imutable_scalar_function(self, fun: ScalarFunction, name):
        if fun is not None and not isinstance(fun, ScalarFunction):
            raise ValueError(f"{name} must be an instance of ScalarFunction")


class VectorPhysicsMixin:
    def _separate_solutions(self, array: Array):
        sol = VectorSolution(self.basis)
        sol.set_values(array)
        sol.set_matrix_jacobian(sol._initial_matrix_jacobian())
        return sol

    def _check_is_imutable_vector_function(self, fun: VectorFunction, name):
        if fun is not None and not isinstance(fun, VectorFunction):
            raise ValueError(f"{name} must be an instance of VectorFunction")


class AdvectionDiffusionReactionEquation(ScalarPhysicsMixin, Physics):
    def __init__(
        self,
        forcing: ScalarFunction = None,
        diffusion: ScalarFunction = None,
        reaction_op: Operator = None,
        velocity_field: VectorFunction = None,
    ):
        self._check_is_imutable_scalar_function(forcing, "forcing")
        self._check_is_imutable_scalar_function(diffusion, "diffusion")
        if reaction_op is not None and not isinstance(reaction_op, Operator):
            raise ValueError("reaction must be an instance of Operator")
        if velocity_field is not None and not isinstance(
            velocity_field, VectorFunction
        ):
            raise ValueError(
                "velocity_field must be an instance of VectorFunction"
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        self._diffusion = diffusion
        self._reaction_op = reaction_op
        self._velocity_field = velocity_field
        self._flux_jacobian_implemented = True

    def residual(self, sol: ScalarSolution):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarSolution")
        residual = 0.0
        if self._forcing is not None:
            residual += self._forcing
        if self._diffusion is not None:
            residual += div(self._diffusion * nabla(sol))
        if self._reaction_op is not None:
            residual += self._reaction_op(sol)
        if self._velocity_field is not None:
            # todo combine with diffusion so div only applied once
            residual -= div(sol * self._velocity_field)
        return residual

    def _flux(self, sol: ScalarOperator):
        flux = 0.0
        if self._diffusion is not None:
            flux += self._diffusion * nabla(sol)
        if self._velocity_field is not None:
            flux -= sol * self._velocity_field
        return flux

    def get_functions(self) -> Dict[str, MatrixOperator]:
        funs = super().get_functions()
        funs["forcing"] = self._forcing
        if self._diffusion is not None:
            funs["diffusion"] = self._diffusion
        if self._reaction_op is not None:
            funs["reaction_op"] = self._reaction_op
        if self._velocity_field is not None:
            funs["velocity_field"] = self._velocity_field
        return funs


class ShallowIceEquation(ScalarPhysicsMixin, Physics):
    def __init__(
        self,
        bed: ScalarFunction,
        friction: ScalarFunction,
        A: float,
        rho: float,
        forcing: ScalarFunction = None,
        eps: float = 0.0,
    ):
        self._check_is_imutable_scalar_function(bed, "bed")
        self._check_is_imutable_scalar_function(friction, "friction")
        self._check_is_imutable_scalar_function(forcing, "forcing")
        super().__init__(forcing.basis)
        self._bed = bed
        self._friction = friction
        self._forcing = forcing
        self._eps = eps
        self._A = A
        self._rho = rho
        self._g = 9.81
        self._n = 3
        self._gamma = (
            2 * self._A * (self._rho * self._g) ** self._n / (self._n + 2)
        )
        self._friction_frac = self._g * self._rho / self._friction
        self._flux_jacobian_implemented = True

    def _flux(self, sol: ScalarOperator):
        surf_grad = nabla(self._bed + sol)
        mon_op = ScalarMonomialOperator(degree=self._n + 2)
        diffusion = (
            self._gamma * mon_op(sol)
            # * ((sqmagnitude(surf_grad) + self._eps) ** (self._n - 1)).sqrt()
            * sqmagnitude(surf_grad)  # true because self._n = 3
            + (self._friction_frac * sol**2)
        )
        return diffusion * surf_grad

    def residual(self, sol: ScalarSolution):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarSolution")
        residual = 0.0
        if self._forcing is not None:
            residual += self._forcing
        residual += div(self._flux(sol))
        return residual

    def get_functions(self) -> Dict[str, MatrixOperator]:
        funs = super().get_functions()
        funs["forcing"] = self._forcing


class HelmholtzEquation(AdvectionDiffusionReactionEquation):
    def __init__(
        self,
        sq_wave_num: ScalarOperator,
        forcing: ScalarFunction = None,
    ):
        basis = sq_wave_num.basis
        diffusion = ScalarFunctionFromCallable(
            basis, lambda x: basis._bkd.ones(x.shape[1])
        )
        reaction_op = ScalarMonomialOperator(degree=1, coef=sq_wave_num)
        super().__init__(
            forcing,
            diffusion,
            reaction_op,
            None,
        )


class ShallowWaveEquation(VectorPhysicsMixin, Physics):
    def __init__(
        self,
        bed: ScalarOperator,
        forcing: ScalarFunction = None,
    ):
        self._forcing = forcing
        self._bed = bed
        self._g = 9.81
        super().__init__(forcing.basis)

    def split_solution(sol: VectorSolution) -> List[ScalarOperator]:
        h = ScalarSolution(sol.get_values()[0, 0, ...])
        uh = ScalarSolution(sol.get_values()[1, 0, ...])
        vh = ScalarSolution(sol.get_values()[2, 0, ...])
        return h, uh, vh

    def _flux(self, sol: VectorSolution):
        h, uh, vh = self.split_solution(sol)
        flux = VectorOperator(
            [
                [uh, vh],
                [uh**2 / h + self._g * h**2, uh * vh / h],
                [uh * vh / h, vh**2 / h + self._g * h**2],
            ]
        )

    def residual(self, sol: VectorSolution):
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        residual = 0.0
        if self._forcing is not None:
            residual += self._forcing
