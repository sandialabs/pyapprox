from abc import abstractmethod, ABC
from typing import Dict
from functools import partial

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.functions import (
    ScalarSolution,
    VectorSolution,
    ScalarFunction,
    ZeroScalarFunction,
    ConstantScalarFunction,
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
    TransientOperatorMixin,
)

from pyapprox.pde.collocation.timeintegration import TransientNewtonResidual
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.pde.collocation.boundaryconditions import (
    BoundaryOperator,
    PeriodicBoundary,
    RobinBoundary,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


class Physics(ABC):
    def __init__(self, basis: OrthogonalCoordinateCollocationBasis):
        self._bkd = basis._bkd
        if not isinstance(basis, OrthogonalCoordinateCollocationBasis):
            raise ValueError(
                "basis must be an instance of "
                "OrthogonalCoordinateCollocationBasis"
            )
        self._basis = basis
        self._flux_jacobian_implemented = False

    def basis(self) -> OrthogonalCoordinateCollocationBasis:
        return self._basis

    def mass_matrix(self, nterms: int) -> Array:
        return self._bkd.eye(nterms)

    def set_boundaries(self, bndrys: list[BoundaryOperator]):
        nperiodic_boundaries = 0
        for bndry in bndrys:
            if isinstance(bndry, PeriodicBoundary):
                nperiodic_boundaries += 1
        # TODO allow physics to set define the minimum number of bndry conditions
        # of certain types, e.g. diffusion equation requires at least one dirichlet
        # boundary (or robin with nonzero solution contribution)
        # if (
        #     len(bndrys) + nperiodic_boundaries
        #     != len(self._basis.mesh._bndrys) * self.ncomponents()
        # ):
        #     raise ValueError("Must set all boundaries")
        self._bndrys = bndrys
        for bndry in self._bndrys:
            if isinstance(bndry, RobinBoundary):
                if not self._flux_jacobian_implemented:
                    raise ValueError(
                        f"RobinBoundary requested but {self} "
                        "does not define _flux_jacobian"
                    )
                bndry.set_flux_functions(
                    partial(self._flux_from_array, bndry._component_id),
                    partial(
                        self._flux_jacobian_from_array, bndry._component_id
                    ),
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
            jac = bndry.apply_to_jacobian(sol_array, jac)
        return jac

    @abstractmethod
    def residual(self, sol: MatrixOperator) -> MatrixOperator:
        raise NotImplementedError

    def _residual_function_from_solution_array(
            self, sol_array: Array
    ) -> MatrixOperator:
        self._bkd.assert_isarray(self._bkd, sol_array)
        sol = self._solution_from_array(sol_array)
        return self.residual(sol)

    def _residual_array_from_solution_array(self, sol_array: Array) -> Array:
        # TODO add option to matrix function that stops jacobian being computed
        # when computing residual. useful for explicit time stepping
        # and linear problems where jacobian does not depend on time or
        # on uncertain parameters of PDE
        res = self._residual_function_from_solution_array(sol_array)
        return res.get_flattened_values()

    def _residual_jacobian_from_solution_array(
            self, sol_array: Array
    ) -> Array:
        res = self._residual_function_from_solution_array(sol_array)
        return res.get_flattened_jacobian()

    @abstractmethod
    def _solution_from_array(self, array: Array) -> MatrixOperator:
        raise NotImplementedError

    def solution_from_array(self, sol_array: Array) -> MatrixOperator:
        sol = self._solution_from_array(sol_array)
        if not isinstance(sol, (ScalarSolution, VectorSolution)):
            raise RuntimeError("sol must be ScalarSolution or VectorSolution")
        return sol

    def _flux(self, sol_array: Array):
        raise NotImplementedError

    def _flux_jacobian_from_array(self, component_id: int, sol_array: Array):
        sol = self.solution_from_array(sol_array)
        flux_jac = self._flux(sol).get_jacobian()
        return flux_jac[:, component_id, :, :]

    def _flux_from_array(self, component_id: int, sol_array: Array):
        sol = self.solution_from_array(sol_array)
        flux = self._flux(sol)
        if flux.ncols() != self.ncomponents():
            raise RuntimeError(
                "flux must be a MatrixFunction with {0} columns".format(
                    self.ncomponents()
                )
            )
        if flux.nrows() != self._basis.nphys_vars():
            raise RuntimeError(
                "flux must be a MatrixFunction with {0} rows".format(
                    self._basis.nphys_vars()
                )
            )
        # return flux.get_values()[:, component_id, :]
        return flux.get_values()[:, component_id, :]

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = dict()
        return funs

    def __repr__(self):
        return "{0}(bndrys={1})".format(self.__class__.__name__, self._bndrys)


class ScalarPhysicsMixin:
    def _solution_from_array(self, array: Array):
        sol = ScalarSolution(self._basis, array)
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
            self._basis, self.ncomponents(), self.ncomponents()
        )
        sol.set_flattened_values(array)
        return sol

    def _check_is_scalar_function(self, fun: ScalarFunction, name):
        if fun is not None and not isinstance(fun, ScalarFunction):
            raise ValueError(f"{name} must be an instance of ScalarFunction")

    def _check_is_vector_function(self, fun: VectorFunction, name):
        if fun is not None and not isinstance(fun, VectorFunction):
            raise ValueError(f"{name} must be an instance of VectorFunction")


class SteadyPhysicsNewtonResidualMixin(NewtonResidual):
    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _linsolve(self, sol_array: Array, res_array: Array) -> Array:
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    def __call__(self, sol_array: Array):
        res_array = self._residual_array_from_solution_array(
            sol_array
        )
        return self.apply_boundary_conditions_to_residual(
            sol_array, res_array
        )

    def _jacobian(self, sol_array: Array):
        # assumes jac called after __call__
        jac = self._residual_jacobian_from_solution_array(sol_array)
        # auto_jac = self._bkd.jacobian(
        #      self._physics._residual_array_from_solution_array,
        #      sol_array
        # )
        # print(self._bkd.abs(jac-auto_jac).max(), "JAC DIFF1")
        # assert self._bkd.allclose(
        #     jac, auto_jac, atol=1e-10, rtol=1e-10
        # )
        return self.apply_boundary_conditions_to_jacobian(
            sol_array, jac
        )


class TransientPhysicsNewtonResidualMixin(TransientNewtonResidual):
    def mass_matrix(self, nterms: int) -> Array:
        return self.mass_matrix(nterms)

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _linsolve(self, sol_array: Array, res_array: Array) -> Array:
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    def __call__(self, sol_array: Array) -> Array:
        self._sol_array = sol_array
        return self._residual_array_from_solution_array(sol_array)

    def jacobian(self, sol_array: Array) -> Array:
        return self._residual_jacobian_from_solution_array(sol_array)

    def _apply_constraints_to_residual(self, res_array: Array) -> Array:
        return self.apply_boundary_conditions_to_residual(
            self._sol_array, res_array
        )

    def _apply_constraints_to_jacobian(self, jac: Array) -> Array:
        return self.apply_boundary_conditions_to_jacobian(
            self._sol_array, jac
        )

    def set_time(self, time: float):
        funs = self.get_functions()
        for name, fun in funs.items():
            if isinstance(fun, TransientOperatorMixin):
                fun.set_time(time)

        for bndry in self._bndrys:
            # TODO CREATE A BASE CLASS WHICH DERIVES CONSTANT AND FUNCTION based bndry
            if hasattr(bndry, "_fun") and isinstance(
                    bndry._fun, TransientOperatorMixin
            ):
                bndry._fun.set_time(time)


class AdvectionDiffusionReactionPhysics(ScalarPhysicsMixin, Physics):
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
        super().__init__(diffusion.basis())
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
            residual += -div(self._flux(sol))
        return residual

    def _flux(self, sol: ScalarSolution):
        # Implement conservative form of flux that includes both diffusion
        # and velocity field.
        # The non conservative version does not include velocity field
        flux = 0.0
        if self._diffusion is not None:
            flux -= self._diffusion * nabla(sol)
        if self._velocity_field is not None:
            flux += sol * self._velocity_field
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


class SteadyAdvectionDiffusionReactionPhysics(
        AdvectionDiffusionReactionPhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


class TransientAdvectionDiffusionReactionPhysics(
        AdvectionDiffusionReactionPhysics, TransientPhysicsNewtonResidualMixin
):
    pass


class ShallowIcePhysics(ScalarPhysicsMixin, Physics):
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
        super().__init__(forcing.basis())
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


class SteadyShallowIcePhysics(
        ShallowIcePhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


# Not yet tested
# class TransientShallowIcePhysics(
#         ShallowIcePhysics, SteadyPhysicsNewtonResidualMixin
# ):
#     pass


class HelmholtzPhysics(AdvectionDiffusionReactionPhysics):
    def __init__(
        self,
        sq_wave_num: ScalarFunction,
        forcing: ScalarFunction = None,
    ):
        basis = sq_wave_num.basis()
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


class SteadyHelmholtzPhysics(
        HelmholtzPhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


class ShallowWavePhysics(VectorPhysicsMixin, Physics):
    def __init__(
        self,
        bed: ScalarOperator,
        forcing: ScalarFunction = None,
    ):
        if forcing is not None and bed.ninput_funs() != forcing.ninput_funs():
            raise ValueError("bed and forcing are inconsistent")
        self._forcing = forcing
        self._bed = bed
        self._g = 9.81
        super().__init__(bed.basis())
        self.set_bed_slope_forcing()
        self._flux_jacobian_implemented = True

    def set_bed_slope_forcing(self):
        self._slope_forcing = VectorOperator(
            self._basis, self.ncomponents(), self.ncomponents()
        )
        zero = ZeroScalarFunction(
            self._basis,
            ninput_funs=self._bed.ninput_funs(),
        )
        slope_gradient_components = nabla(self._bed).get_components()
        self._slope_forcing.set_components([zero] + slope_gradient_components)
        self._slope_forcing *= self._g

    def ncomponents(self) -> int:
        return self._bed.nphys_vars() + 1

    def _flux(self, sol: VectorSolution) -> MatrixOperator:
        # flux is actually a diagonal 3D tensor but store as a matrix
        # because divergence will be applied store flux for each equation
        # as a column
        flux = MatrixOperator(
            sol.basis(),
            sol.ninput_funs(),
            sol.nphys_vars(),
            sol.nrows(),
        )
        if sol.basis().nphys_vars() == 1:
            h, uh = sol.get_components()
            if self._bkd.any(h.get_values() <= 0):
                raise RuntimeError(
                    f"Depth became negative {h.get_values().min()}"
                )
            components = [[uh, uh**2 / h + (0.5 * self._g) * h**2]]
        else:
            h, uh, vh = sol.get_components()
            if self._bkd.any(h.get_values() <= 0):
                raise RuntimeError(
                    f"Depth became negative {h.get_values().min()}"
                )
            uvh = uh * vh / h
            g_hsq = (0.5 * self._g) * h**2
            components = [
                [uh, uh**2 / h + g_hsq, uvh],
                [vh, uvh, vh**2 / h + g_hsq],
            ]
        flux.set_components(components)
        return flux

    def residual(self, sol: VectorSolution) -> VectorOperator:
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        flux = self._flux(sol)
        residual = -div(flux)
        h = sol.get_components()[0]
        residual -= h * self._slope_forcing
        if self._forcing is not None:
            residual += self._forcing
        return residual

    def get_functions(self) -> Dict:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing
        funs["bed"] = self._bed
        return funs


class TransientShallowWavePhysics(
        ShallowWavePhysics, TransientPhysicsNewtonResidualMixin
):
    pass


class TwoSpeciesReactionDiffusionPhysics(VectorPhysicsMixin, Physics):
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
        super().__init__(forcing.basis())
        self._forcing = forcing
        diff_components = diffusion.get_components()
        self._diffusion = MatrixOperator(diffusion.basis(), 2, 2, 2)
        zero = ZeroScalarFunction(
            self._basis,
            ninput_funs=2,
        )
        self._diffusion.set_components(
            [[diff_components[0], zero], [zero, diff_components[1]]]
        )
        self._reaction_op = reaction_op
        self._flux_jacobian_implemented = True

    def ncomponents(self) -> int:
        return 2

    def residual(self, sol: VectorSolution):
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        residual = -div(self._flux(sol))
        residual += self._reaction_op(sol)
        if self._forcing is not None:
            residual += self._forcing
        return residual

    def _flux(self, sol: VectorSolution) -> MatrixOperator:
        flux = -self._diffusion @ vector_nabla(sol)
        return flux.T

    def get_functions(self) -> Dict[str, ScalarOperator]:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing
        funs["diffusion"] = self._diffusion
        funs["reaction_op"] = self._reaction_op
        return funs


class SteadyTwoSpeciesReactionDiffusionPhysics(
        TwoSpeciesReactionDiffusionPhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


class TransientTwoSpeciesReactionDiffusionPhysics(
        TwoSpeciesReactionDiffusionPhysics,
        TransientPhysicsNewtonResidualMixin,
):
    pass


class FitzHughNagumoReactionOperation(VectorOperatorOperation):
    # equations obtained from
    # https://simula-sscp.github.io/SSCP_2024_lectures/L13%20%28FEniCS%20Electrophysiology%29/L05%20-%20FitzhughNagumo-Solution.html
    # see also https://visualpde.com/mathematical-biology/fitzhugh-nagumo.html
    # first state represents voltage across a neuron
    def __init__(
        self,
        coefs: Array,
    ):
        self.set_coefficients(coefs)

    def set_coefficients(self, coefs):
        self._coefs = coefs

    def __call__(self, sol: VectorSolution):
        alpha, eps, beta, gamma = self._coefs
        u0, u1 = sol.get_components()
        vec = VectorOperator(sol.basis(), sol.ninput_funs(), sol.nrows())
        vec.set_components(
            [u0 * (1. - u0)*(u0 - alpha) - u1, eps * (beta * u0 - gamma * u1)]
        )
        return vec


class FitzHughNagumoPhysics(TwoSpeciesReactionDiffusionPhysics):
    def __init__(
        self,
        basis,
        forcing: VectorFunction = None,
    ):
        diffusion_components = [
            ConstantScalarFunction(basis, 1e-3, 2),
            # ConstantScalarFunction(basis, 1e-4, 2),
            ZeroScalarFunction(basis, 2),
            # ConstantScalarFunction(basis, 1e-2, 2),
        ]
        diffusion = VectorFunction(basis, 2, 2)
        diffusion.set_components(diffusion_components)
        # set coefs to nominal values
        coefs = basis._bkd.array([0.1, 0.01, 0.5, 1.0])
        reaction_op = FitzHughNagumoReactionOperation(coefs)
        super().__init__(forcing, diffusion, reaction_op)

    def set_coefficients(self, coefs):
        self._reaction_op.set_coefficients(coefs)


class TransientFitzHughNagumoPhysics(
        FitzHughNagumoPhysics, TransientPhysicsNewtonResidualMixin
):
    pass


class ShallowShelfVelocityPhysics(VectorPhysicsMixin, Physics):
    def __init__(
        self,
        depth: ScalarFunction,
        bed: ScalarFunction,
        friction: ScalarFunction,
        A: float,
        rho: float,
        velocity_forcing: VectorFunction = None,
    ):
        self._check_is_scalar_function(bed, "bed")
        self._check_is_scalar_function(friction, "friction")
        self._check_is_vector_function(velocity_forcing, "velocity_forcing")
        self.set_depth(depth)
        super().__init__(depth.basis())
        self._bed = bed
        self._friction = friction
        self._velocity_forcing = velocity_forcing

        self._A = A
        self._rho = rho
        self._g = 9.81
        self._n = 3
        self._fixed_strain_rate = None

        # for now assume this phsyics is only used to solve steady state
        # problem, so depth does not change
        surf_grad = nabla(self._bed + self._depth)
        self._weighted_surf_grad = (self._rho * self._g) * surf_grad
        self._const = 0.5 * self._A ** (-1 / self._n)

        self._flux_jacobian_implemented = True

    def set_depth(self, depth: ScalarOperator):
        if not isinstance(depth, ScalarOperator):
            raise ValueError("depth must be an instance of ScalarOperator")
        self._depth = depth

    def _effective_strain_rate(self, ux, uy, vx, vy):
        # add regulization term to avoid numeric zero
        return (
            ux**2 + vy**2 + ux * vy + 0.25 * (uy + vx) ** 2 + 1e-10
        ) ** (0.5)

    def _fix_strain_rate(self, strain_rate: ScalarFunction):
        self._fixed_strain_rate = strain_rate

    def _unfix_strain_rate(self):
        self._fixed_strain_rate = None

    def _get_strain_rate_from_solution_array(
            self, sol_array: Array
    ) -> ScalarOperator:
        # Must be VectorFunction (not VectorSolution) so its jacobian is zero
        sol = VectorFunction(
            self._basis, self.ncomponents(), self.ncomponents()
        )
        sol.set_flattened_values(sol_array)
        u, v = sol.get_components()
        ux = u.deriv(0)
        uy = u.deriv(1)
        vx = v.deriv(0)
        vy = v.deriv(1)
        return self._effective_strain_rate(ux, uy, vx, vy)

    def _flux(self, sol: VectorSolution) -> MatrixOperator:
        strain_tensor = MatrixOperator(sol.basis(), 2, 2, 2)
        u, v = sol.get_components()
        ux = u.deriv(0)
        uy = u.deriv(1)
        vx = v.deriv(0)
        vy = v.deriv(1)
        offdiag = 0.5 * (uy + vx)
        strain_tensor_components = [
            [2.0 * ux + vy, offdiag],
            [offdiag, ux + 2.0 * vy],
        ]
        strain_tensor.set_components(strain_tensor_components)
        rate = self._effective_strain_rate(ux, uy, vx, vy)
        if self._fixed_strain_rate is not None:
            # used for picard iteration
            # print(
            #     self._fixed_strain_rate.get_values().min(),
            #     self._fixed_strain_rate.get_values().max(),
            #     self._bkd.norm(self._fixed_strain_rate.get_values())
            # )
            mu = self._const * self._fixed_strain_rate ** (1 / self._n - 1)
        else:
            mu = self._const * rate ** (1 / self._n - 1)
        flux = 2.0 * mu * self._depth * strain_tensor
        return flux

    def residual(self, sol: VectorSolution) -> VectorOperator:
        #print(sol.get_values().requires_grad)
        # print(self._friction.get_values().requires_grad, 'friction')
        flux = self._flux(sol)
        residual = (
            div(flux)
            - self._friction * sol
            - self._depth * self._weighted_surf_grad
        )
        if self._velocity_forcing is not None:
            residual += self._velocity_forcing
        return residual

    def ncomponents(self):
        return 2

    def get_functions(self) -> Dict:
        funs = super().get_functions()
        if self._velocity_forcing is not None:
            funs["velocity_forcing"] = self._velocity_forcing
        funs["bed"] = self._bed
        funs["friction"] = self._friction
        return funs


class SteadyShallowShelfVelocityPhysics(
        ShallowShelfVelocityPhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


class ShallowShelfDepthPhysics(ScalarPhysicsMixin, Physics):
    def __init__(self, depth_forcing: ScalarFunction = None):
        self._check_is_scalar_function(depth_forcing, "depth_forcing")
        super().__init__(depth_forcing.basis())
        self._depth_forcing = depth_forcing
        self._flux_jacobian_implemented = True

    def set_velocities(self, velocities: VectorSolution):
        self._velocities = velocities

    def _flux(self, sol: ScalarSolution):
        # sol = depth
        return sol * self._velocities

    def residual(self, sol: ScalarSolution) -> ScalarOperator:
        residual = -div(self._flux(sol))
        if self._depth_forcing is not None:
            residual += self._depth_forcing
        return residual

    def get_functions(self) -> Dict:
        funs = super().get_functions()
        if self._depth_forcing is not None:
            funs["depth_forcing"] = self._depth_forcing
        return funs


class SplitPhysicsMixin(ABC):
    @abstractmethod
    def _transient_physics_solution_from_array(
            self, sol_array: Array
    ) -> MatrixOperator:
        raise NotImplementedError

    @abstractmethod
    def mass_matrix(self, nterms: int) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _steady_physics_solution_from_array(
            self, sol_array: Array
    ) -> MatrixOperator:
        raise NotImplementedError

    def _residual_array_from_solution_array(
            self, transient_sol_array: Array
    ) -> Array:
        # this will be called by TimeIntegratorNewtonResidual which will only
        # pass in transient_sol array. ignore this input and
        # just use self._sol_array which contains transient_sol components
        # and steady components
        sol = self._transient_physics_solution_from_array(self._sol_array)
        return self._transient_physics.residual(sol).get_flattened_values()

    def _residual_jacobian_from_solution_array(
            self, transient_sol_array: Array
    ):
        # this will be called by TimeIntegratorNewtonResidual
        sol = self._transient_physics_solution_from_array(self._sol_array)
        return self._transient_physics.residual(sol).get_flattened_jacobian()

    def steady_value(self, sol_array: Array) -> Array:
        sol = self._steady_physics_solution_from_array(sol_array)
        return self._steady_physics.residual(sol).get_flattened_values()

    def steady_jacobian(self, sol_array: Array) -> Array:
        sol = self._steady_physics_solution_from_array(sol_array)

        return self._steady_physics.residual(sol).get_flattened_jacobian()

    def _set_steady_and_transient_components(self, sol_array: Array):
        self._sol_array = sol_array
        sol = self._solution_from_array(sol_array)
        self._set_transient_physics_components(sol)
        self._set_steady_physics_components(sol)

    @abstractmethod
    def _set_transient_physics_components(self, sol: VectorSolution):
        raise NotImplementedError

    @abstractmethod
    def _set_steady_physics_components(self, sol: VectorSolution):
        raise NotImplementedError


class TransientSplitPhysicsNewtonResidualMixin(
        TransientPhysicsNewtonResidualMixin
):
    def __call__(self, sol_array: Array) -> Array:
        # do not set self._sol_array as done in base class
        # self._sol_array = sol_array
        return self._residual_array_from_solution_array(sol_array)


class ShallowShelfDepthVelocityPhysics(
        SplitPhysicsMixin, VectorPhysicsMixin, Physics
):
    def __init__(
        self,
        bed: ScalarFunction,
        friction: ScalarFunction,
        A: float,
        rho: float,
        depth_forcing: ScalarFunction = None,
        velocity_forcing: VectorFunction = None,
    ):
        super().__init__(bed.basis())
        # initialize depth to zero, it will be overwritten later
        depth = ZeroScalarFunction(self._basis, self.ncomponents())
        self._flux_jacobian_implemented = True
        self._steady_physics = ShallowShelfVelocityPhysics(
            depth, bed, friction, A, rho, velocity_forcing
        )
        self._transient_physics = ShallowShelfDepthPhysics(depth_forcing)

    def ncomponents(self) -> int:
        return self._basis.nphys_vars()+1

    def mass_matrix(self, nterms: int) -> Array:
        return self._bkd.hstack(
            (self._bkd.eye(nterms), self._bkd.zeros((nterms, 2*nterms)))
        )

    def _extract_velocities_components(
            self, sol: VectorSolution
    ) -> VectorSolution:
        u, v = sol.get_components()[1:]
        velocities = VectorSolution(
            self._basis, self.ncomponents(), self.ncomponents()-1
        )
        velocities.set_components([u, v])
        return velocities

    def _set_transient_physics_components(self, sol: VectorSolution):
        velocities = self._extract_velocities_components(sol)
        self._transient_physics.set_velocities(velocities)

    def _set_steady_physics_components(self, sol: VectorSolution):
        depth = sol.get_components()[0]
        self._steady_physics.set_depth(depth)

    def _transient_physics_solution_from_array(
            self, sol_array: Array
    ) -> MatrixOperator:
        sol = self._solution_from_array(sol_array)
        depth = sol.get_components()[0]
        self._set_transient_physics_components(sol)
        return depth

    def _steady_physics_solution_from_array(
            self, sol_array: Array
    ) -> MatrixOperator:
        sol = self._solution_from_array(sol_array)
        velocities = self._extract_velocities_components(sol)
        self._set_steady_physics_components(sol)
        return velocities

    def residual(self, sol: MatrixOperator) -> MatrixOperator:
        # depth, u, v = sol.get_components()
        # must be MatrixOperator not VectorOperator even though it has
        # only 1 column because every other physics does the same thing
        residual = MatrixOperator(
            self._basis, self.ncomponents(), self.ncomponents(), 1
        )
        depth = self._transient_physics_solution_from_array(
            sol.get_flattened_values()
        )
        velocities = self._steady_physics_solution_from_array(
            sol.get_flattened_values()
        )
        depth_residual = self._transient_physics.residual(depth)
        velocities_residual = self._steady_physics.residual(velocities)
        residual.set_components(
            [[depth_residual]]
            + [v for v in velocities_residual.get_components()]
        )
        return residual

    def _flux(self, sol: VectorSolution) -> MatrixOperator:
        depth = self._transient_physics_solution_from_array(
            sol.get_flattened_values()
        )
        transient_flux = self._transient_physics._flux(depth)
        velocities = self._steady_physics_solution_from_array(
            sol.get_flattened_values()
        )
        steady_flux = self._steady_physics._flux(velocities)
        flux = MatrixOperator(
            self._basis,
            self.ncomponents(),
            self._basis.nphys_vars(),
            self.ncomponents(),
        )
        transient_flux_components = transient_flux.get_components()
        steady_flux_components = steady_flux.get_components()
        flux_components = [
            transient_flux_components[ii] + steady_flux_components[ii]
            for ii in range(self._basis.nphys_vars())
        ]
        flux.set_components(flux_components)
        return flux

    def get_functions(self) -> Dict:
        return {
            **self._steady_physics.get_functions(),
            **self._transient_physics.get_functions()
        }


class TransientShallowShelfDepthVelocityPhysics(
        ShallowShelfDepthVelocityPhysics,
        TransientSplitPhysicsNewtonResidualMixin,
):
    pass


class Isotropic2DLinearElasticityPhysics(VectorPhysicsMixin, Physics):
    def __init__(
        self,
        lamda: ScalarFunction,
        mu: ScalarFunction,
        forcing: VectorFunction = None,
    ):
        self._check_is_scalar_function(lamda, "lambda")
        self._check_is_scalar_function(mu, "mu")
        self._check_is_vector_function(forcing, "forcing")
        self._lambda = lamda
        self._mu = mu
        self._forcing = forcing
        super().__init__(mu.basis())
        self._flux_jacobian_implemented = True

    def ncomponents(self) -> int:
        return 2

    def _flux(self, sol: VectorSolution) -> MatrixOperator:
        u, v = sol.get_components()
        ux = u.deriv(0)
        uy = u.deriv(1)
        vx = v.deriv(0)
        vy = v.deriv(1)

        exx = ux
        exy = 0.5 * (vx+uy)
        eyy = vy

        # no need to store strain tensor as a matrix operator
        # strain_tensor = MatrixOperator(sol.basis(), 2, 2, 2)
        # strain_tensor_components = [
        #     [exx, exy]
        #     [exy, eyy],
        # ]
        # strain_tensor.set_components(strain_tensor_components)

        trace_tensor = exx + eyy
        two_mu = 2. * self._mu
        tauxx = self._lambda * trace_tensor + two_mu * exx
        tauxy = two_mu * exy
        tauyy = self._lambda * trace_tensor + two_mu * eyy

        flux = MatrixOperator(sol.basis(), 2, 2, 2)
        flux_components = [
            [tauxx, tauxy],
            [tauxy, tauyy],
        ]
        flux.set_components(flux_components)
        return flux

    def residual(self, sol: VectorSolution):
        if not isinstance(sol, VectorSolution):
            raise ValueError("sol must be an instance of VectorSolution")
        residual = div(self._flux(sol))
        if self._forcing is not None:
            residual += self._forcing
        return residual

    def get_functions(self) -> Dict:
        funs = super().get_functions()
        if self._forcing is not None:
            funs["forcing"] = self._forcing
        funs["lambda"] = self._lambda
        funs["mu"] = self._mu
        return funs


class SteadyIsotropic2DLinearElasticityPhysics(
        Isotropic2DLinearElasticityPhysics, SteadyPhysicsNewtonResidualMixin
):
    pass


# Not Yet Tested
# class TransientIsotropic2DLinearElasticityPhysics(
#         Isotropic2DLinearElasticityPhysics,
#         TransientPhysicsNewtonResidualMixin
# ):
#     pass
