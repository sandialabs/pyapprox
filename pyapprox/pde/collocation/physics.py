from abc import abstractmethod
from typing import Dict, List

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.functions import (
    ScalarSolution,
    VectorSolution,
    ScalarFunction,
    VectorFunction,
    ScalarOperator,
    ScalarOperatorOperation,
    ScalarFunctionFromCallable,
    ScalarMonomialOperator,
    VectorOperator,
    VectorOperatorOperation,
    MatrixOperator,
    nabla,
    vector_nabla,
    div,
)
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.pde.collocation.boundaryconditions import (
    BoundaryOperator,
    PeriodicBoundary,
    RobinBoundary,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


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
        if (
            len(bndrys) + nperiodic_boundaries
            != len(self.basis.mesh._bndrys) * self.ncomponents()
        ):
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
    def residual(self, sol: ScalarOperator):
        raise NotImplementedError

    def _residual_function_from_solution_array(self, sol_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        sol = self._solution_from_array(sol_array)
        return self.residual(sol)

    def _residual_array_from_solution_array(self, sol_array: Array):
        # TODO add option to matrix function that stops jacobian being computed
        # when computing residual. useful for explicit time stepping
        # and linear problems where jacobian does not depend on time or
        # on uncertain parameters of PDE
        res = self._residual_function_from_solution_array(sol_array)
        return res.get_flattened_values()

    def _residual_jacobian_from_solution_array(self, sol_array: Array):
        res = self._residual_function_from_solution_array(sol_array)
        return res.get_flattened_jacobian()

    @abstractmethod
    def _solution_from_array(self, array):
        raise NotImplementedError

    def solution_from_array(self, sol_array: Array):
        sol = self._solution_from_array(sol_array)
        if not isinstance(sol, (ScalarSolution, VectorSolution)):
            raise RuntimeError("sol must be ScalarSolution or VectorSolution")
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
        sol = self.solution_from_array(sol_array)
        flux_jac = self._flux(sol).get_jacobian()
        return flux_jac[:, 0, :, :]

    def _flux_from_array(self, sol_array: Array):
        sol = self.solution_from_array(sol_array)
        flux = self._flux(sol)
        if flux.ncols() != 1:
            raise RuntimeError("flux must be a MatrixFunction with one column")
        return flux.get_values()[:, 0, :]

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = dict()
        return funs

    def __repr__(self):
        return "{0}(bndrys={1})".format(self.__class__.__name__, self._bndrys)


class ScalarPhysicsMixin:
    def _solution_from_array(self, array: Array):
        sol = ScalarSolution(self.basis, array)
        return sol

    def _check_is_scalar_function(self, fun: ScalarFunction, name):
        if fun is not None and not isinstance(fun, ScalarFunction):
            raise ValueError(f"{name} must be an instance of ScalarFunction")

    def ncomponents(self) -> int:
        return 1


class VectorPhysicsMixin:
    def ncomponents(self):
        raise NotImplementedError

    def _solution_from_array(self, array: Array):
        sol = VectorSolution(
            self.basis, self.ncomponents(), self.ncomponents()
        )
        sol.set_flattened_values(array)
        return sol

    def _check_is_vector_function(self, fun: VectorFunction, name):
        if fun is not None and not isinstance(fun, VectorFunction):
            raise ValueError(f"{name} must be an instance of VectorFunction")


class AdvectionDiffusionReactionEquation(ScalarPhysicsMixin, Physics):
    def __init__(
        self,
        forcing: ScalarFunction = None,
        diffusion: ScalarFunction = None,
        reaction_op: ScalarOperatorOperation = None,
        velocity_field: VectorFunction = None,
    ):
        self._check_is_scalar_function(forcing, "forcing")
        self._check_is_scalar_function(diffusion, "diffusion")
        if reaction_op is not None and not isinstance(
            reaction_op, ScalarOperatorOperation
        ):
            raise ValueError(
                "reaction must be an instance of ScalarOperatorOperation"
            )
        if velocity_field is not None and not isinstance(
            velocity_field, VectorFunction
        ):
            raise ValueError(
                "velocity_field {0} must be an instance of VectorFunction".format(
                    velocity_field
                )
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        self._diffusion = diffusion
        self._reaction_op = reaction_op
        self._velocity_field = velocity_field
        self._flux_jacobian_implemented = (
            self._velocity_field is not None or self._diffusion is not None
        )

    def residual(self, sol: ScalarSolution):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarSolution")
        residual = 0.0
        if self._forcing is not None:
            residual += self._forcing
        if self._reaction_op is not None:
            residual += self._reaction_op(sol)
        if self._flux_jacobian_from_array:
            residual += div(self._flux(sol))
        return residual

    def _flux(self, sol: ScalarSolution):
        flux = 0.0
        if self._diffusion is not None:
            flux += self._diffusion * nabla(sol)
        if self._velocity_field is not None:
            flux -= sol * self._velocity_field
        return flux

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = super().get_functions()
        if self._forcing is not None:
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
        self._check_is_scalar_function(bed, "bed")
        self._check_is_scalar_function(friction, "friction")
        self._check_is_scalar_function(forcing, "forcing")
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

    def _flux(self, sol: ScalarSolution):
        surf_grad = nabla(self._bed + sol)
        mon_op = ScalarMonomialOperator(degree=self._n + 2)
        diffusion = (
            self._gamma * mon_op(sol)
            # * ((sqmagnitude(surf_grad) + self._eps) ** ((self._n - 1) / 2)
            * surf_grad.sqnorm()  # true because self._n = 3
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

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing


class HelmholtzEquation(AdvectionDiffusionReactionEquation):
    def __init__(
        self,
        sq_wave_num: ScalarFunction,
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

    def ncomponents(self) -> int:
        return self._bed.nphys_vars() + 1

    def _flux(self, sol: VectorSolution):
        # flux is actually a diagonal 3D tensor but store as a matrix
        # because divergence will be applied store flux for each equation
        # as a column
        flux = MatrixOperator(
            sol.basis,
            sol.ninput_funs(),
            sol.nphys_vars(),
            sol.nrows(),
        )
        if sol.basis.nphys_vars() == 1:
            h, uh = sol.get_components()
            components = [[uh, uh**2 / h + (0.5 * self._g) * h**2]]
        else:
            h, uh, vh = sol.get_components()
            uvh = uh * vh
            g_hsq = (0.5 * self._g) * h**2
            components = [
                [uh, uh**2 / h + g_hsq, uvh],
                [vh, uvh, vh**2 / h + g_hsq],
            ]
        flux.set_components(components)
        return -flux

    def residual(self, sol: VectorSolution):
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        flux = self._flux(sol)
        residual = div(flux)
        if self._forcing is not None:
            residual += self._forcing
        return residual

    def get_functions(self) -> Dict:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing
        funs["bed"] = self._bed
        return funs


class TwoSpeciesReactionDiffusionEquations(VectorPhysicsMixin, Physics):
    def __init__(
        self,
        forcing: VectorFunction = None,
        diffusion: VectorFunction = None,
        reaction_op: VectorOperatorOperation = None,
    ):
        self._check_is_vector_function(forcing, "forcing")
        self._check_is_vector_function(diffusion, "diffusion")
        if not isinstance(reaction_op, VectorOperatorOperation):
            raise ValueError(
                "reaction must be an instance of VectorOperatorOperation"
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        diff_components = diffusion.get_components()
        self._diffusion = MatrixOperator(
            diffusion.basis, 2, 2, 2
        )
        zero = ScalarFunction(
            self.basis,
            self._bkd.zeros(self.basis.mesh.nmesh_pts()),
            ninput_funs=2,
        )
        self._diffusion.set_components(
            [[diff_components[0], zero],
             [zero, diff_components[1]]]
        )
        self._reaction_op = reaction_op
        self._flux_jacobian_implemented = True

    def ncomponents(self) -> int:
        return 2

    def residual(self, sol: VectorSolution):
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        residual = div(self._flux(sol).T)
        residual += self._reaction_op(sol)
        if self._forcing is not None:
            residual += self._forcing
        return residual

    def _flux(self, sol: VectorSolution):
        flux = self._diffusion @ vector_nabla(sol)
        return flux

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing
        funs["diffusion"] = self._diffusion
        funs["reaction_op"] = self._reaction_op
        return funs
