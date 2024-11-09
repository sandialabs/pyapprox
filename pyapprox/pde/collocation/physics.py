from abc import abstractmethod
from typing import Dict

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.functions import (
    MatrixFunction,
    ScalarSolution,
    ImutableScalarFunction,
    ScalarFunction,
    ImutableVectorFunction,
    SolutionMixin,
    Operator,
)
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.pde.collocation.boundaryconditions import (
    BoundaryFunction,
    PeriodicBoundary,
    RobinBoundary,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis
from pyapprox.pde.collocation.functions import nabla, div


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

    def set_boundaries(self, bndrys: list[BoundaryFunction]):
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
                bndry.set_flux_jacobian(self._flux_jacobian)

    def apply_boundary_conditions_to_residual(
        self, sol_array: Array, res_array: Array, jac: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_residual(sol_array, res_array, jac)
        return res_array

    def apply_boundary_conditions_to_jacobian(
        self, sol_array: Array, jac: Array
    ):
        for bndry in self._bndrys:
            res_array = bndry.apply_to_jacobian(sol_array, jac)
        return res_array

    @abstractmethod
    def residual(self, sol: MatrixFunction):
        raise NotImplementedError

    def _residual_function_from_solution_array(self, sol_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        sol = self._separate_solutions(sol_array)
        return self.residual(sol)

    def _residual_array_and_jacobian_from_solution_array(
        self, sol_array: Array
    ):
        res = self._residual_function_from_solution_array(sol_array)
        res_array = self._bkd.flatten(res.get_values())
        jac = res.get_matrix_jacobian()
        jac = self._bkd.reshape(
            jac, (jac.shape[0] * jac.shape[2], jac.shape[3])
        )
        return res_array, jac

    def _residual_array_from_solution_array(self, sol_array: Array):
        # TODO add option to matrix function that stops jacobian being computed
        # when computing residual. useful for explicit time stepping
        # and linear problems where jacobian does not depend on time or
        # on uncertain parameters of PDE
        return self._residual_array_and_jacobian_from_solution_array(
            sol_array
        )[0]

    @abstractmethod
    def _separate_solutions(self, array):
        raise NotImplementedError

    def separate_solutions(self, sol_array: Array):
        sol = self._separate_solutions(sol_array)
        if not isinstance(sol, SolutionMixin) or not isinstance(
            sol, MatrixFunction
        ):
            raise RuntimeError(
                "sol must be derived from SolutionMixin and MatrixFunction"
            )
        return sol

    def __call__(self, sol_array: Array):
        res_array, jac = self._residual_array_and_jacobian_from_solution_array(
            sol_array
        )
        self._jac = jac
        return self.apply_boundary_conditions_to_residual(
            sol_array, res_array, jac
        )

    def jacobian(self, sol_array: Array):
        # assumes jac called after __call__
        return self.apply_boundary_conditions_to_jacobian(sol_array, self._jac)

    def _linsolve(self, sol_array: Array, res_array: Array):
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _flux_jacobian(self, sol_array: Array):
        raise NotImplementedError

    def get_functions(self) -> Dict[str, MatrixFunction]:
        return self._bndrys

    def __repr__(self):
        return "{0}(bndrys={1})".format(self.__class__.__name__, self._bndrys)


class ScalarPhysicsMixin:
    def _separate_solutions(self, array: Array):
        sol = ScalarSolution(self.basis)
        sol.set_values(array)
        sol.set_matrix_jacobian(sol._initial_matrix_jacobian())
        return sol


class AdvectionDiffusionReactionEquation(ScalarPhysicsMixin, Physics):
    def __init__(
        self,
        forcing: ImutableScalarFunction = None,
        diffusion: ImutableScalarFunction = None,
        reaction_op: Operator = None,
        velocity_field: ImutableVectorFunction = None,
    ):
        if forcing is not None and not isinstance(
                forcing, ImutableScalarFunction
        ):
            raise ValueError(
                "forcing must be an instance of ImutableScalarFunction"
            )
        if diffusion is not None and not isinstance(
                diffusion, ImutableScalarFunction
        ):
            raise ValueError(
                "diffusion must be an instance of ImutableScalarFunction"
            )
        if reaction_op is not None and not isinstance(reaction_op, Operator):
            raise ValueError(
                "reaction must be an instance of Operator"
            )
        if velocity_field is not None and not isinstance(
                velocity_field, ImutableVectorFunction
        ):
            raise ValueError(
                "velocity_field must be an instance of ImutableVectorFunction"
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        self._diffusion = diffusion
        self._reaction_op = reaction_op
        self._velocity_field = velocity_field
        self._flux_jacobian_implemented = True

    def residual(self, sol: ScalarFunction):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarFunction")
        residual = 0.
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

    def _flux_jacobian(self, sol_array: Array):
        sol = self.separate_solutions(sol_array)
        flux_jac = 0
        if self._diffusion is not None:
            flux_jac += (self._diffusion * nabla(sol)).get_matrix_jacobian()
        if self._velocity_field is not None:
            flux_jac -= (sol * self._velocity_field).get_matrix_jacobian()
        return flux_jac[:, 0, :, :]

    def get_functions(self) -> Dict[str, MatrixFunction]:
        funs = super().get_functions()
        funs["forcing"] = self._forcing,
        if self._diffusion is not None:
            funs["diffusion"] = self._diffusion
        if self._reaction is not None:
            funs["reaction_op"] = self._reaction_op
        if self._advection is not None:
            funs["advection"] = self._advection
        return funs
