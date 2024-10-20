from abc import abstractmethod

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.pde.collocation.functions import (
    MatrixFunction,
    ScalarSolution,
    ImutableScalarFunction,
    ScalarFunction,
    ImutableVectorFunction,
    SolutionMixin,
)
from pyapprox.pde.collocation.newton import NewtonResidual
from pyapprox.pde.collocation.boundaryconditions import (
    BoundaryFunction,
    PeriodicBoundary,
    RobinBoundary,
)
from pyapprox.pde.collocation.basis import OrthogonalCoordinateCollocationBasis


# nabla f(u) = [D_1f_1,    0  ], d/du (nabla f(u)) = [D_1f_1'(u),     0     ]
#            = [  0   , D_2f_2]                      [   0      , D_2 f'(u) ]
# where f'(u) = d/du f(u)
def nabla(fun: MatrixFunction):
    """Gradient of a scalar valued function"""
    funvalues = fun.get_matrix_values()[0, 0]
    fun_jac = fun.get_matrix_jacobian()
    # todo need to create 3d array
    grad_vals = fun._bkd.stack(
        [
            fun.basis._deriv_mats[dd] @ funvalues
            for dd in range(fun.nphys_vars())
        ],
        axis=0,
    )[:, None, :]
    grad_jacs = fun._bkd.stack(
        [
            (fun.basis._deriv_mats[dd] @ fun_jac[0, 0])[None, :]
            for dd in range(fun.nphys_vars())
        ],
        axis=0,
    )
    return MatrixFunction(fun.basis, fun.nphys_vars(), 1, grad_vals, grad_jacs)


# div f = [D_1 f_1(u) + D_2f_2(u)],  (div f)' = [D_1f'_1(u) + D_2f'_2(u)]
def div(fun: MatrixFunction):
    """Divergence of a vector valued function."""
    if fun._ncols != 1:
        raise ValueError("Fun must be a vector valued function")
    fun_values = fun.get_values()[:, 0, ...]
    fun_jacs = fun.get_matrix_jacobian()[:, 0, ...]
    dmats = fun._bkd.stack(fun.basis._deriv_mats, axis=0)
    # dmats: (nrows, n, n)
    # fun_values : (nrows, n)
    div_vals = fun._bkd.sum(
        fun._bkd.einsum("ijk,ik->ij", dmats, fun_values), axis=0
    )
    # dmats: (nrows, n, n)
    # fun_jacs : (nrows, n, n)
    div_jac = fun._bkd.sum(
        fun._bkd.einsum("ijk,ikm->ijm", dmats, fun_jacs), axis=0
    )
    return MatrixFunction(
        fun.basis, 1, 1, div_vals[None, None, :], div_jac[None, None, ...]
    )


# div (nabla f)  = [D_1, D_2][D_1f_1,    0  ] = [D_1D_1f_1,    0     ]
#                            [  0   , D_2f_2] = [  0      , D_2D_2f_2]
# d/du (nabla f(u)) = [D_1D_1f_1'(u),     0        ]
#                     [   0      ,    D_2D_2 f'(u) ]
def laplace(fun: MatrixFunction):
    """Laplacian of a scalar valued function"""
    return div(nabla(fun))


def fdotgradf(fun: MatrixFunction):
    r"""(f \cdot nabla f)f of a vector-valued function f"""
    pass


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

    def linsolve(self, sol_array: Array, res_array: Array):
        self._bkd.assert_isarray(self._bkd, sol_array)
        self._bkd.assert_isarray(self._bkd, res_array)
        return self._linsolve(sol_array, res_array)

    def _flux_jacobian(self, sol_array: Array):
        raise NotImplementedError


class LinearPhysicsMixin:
    # TODO add option of prefactoring jacobian
    # def _linsolve(self, sol_array: Array, res_array: Array):
    #     return qr_solve(self._Q, self._R, res_array, bkd=self._bkd)

    def _linsolve(self, sol_array: Array, res_array: Array):
        return self._bkd.solve(self.jacobian(sol_array), res_array)

    # def _residual_from_array(self, sol_array: Array):
    #     # Only compute linear jacobian once.
    #     # This is useful for transient problems or for steady state
    #     # parameterized PDEs with jacobians that are not dependent on
    #     # the uncertain parameters
    #     if hasattr(self, "_linear_jac"):
    #         return self._linear_jac @ sol_array
    #     return self._linear_residual_from_function().get_values()


class NonLinearPhysicsMixin:
    def _linsolve(self, sol_array: Array, res_array: Array):
        return self._bkd.solve(self.jacobian(sol_array), res_array)


class ScalarPhysicsMixin:
    def _separate_solutions(self, array: Array):
        sol = ScalarSolution(self.basis)
        sol.set_values(array)
        sol.set_matrix_jacobian(sol._initial_matrix_jacobian())
        return sol


class LinearDiffusionEquation(LinearPhysicsMixin, ScalarPhysicsMixin, Physics):
    def __init__(
        self,
        forcing: ImutableScalarFunction,
        diffusion: ImutableScalarFunction,
    ):
        if not isinstance(forcing, ImutableScalarFunction):
            raise ValueError(
                "forcing must be an instance of ImutableScalarFunction"
            )
        if not isinstance(diffusion, ImutableScalarFunction):
            raise ValueError(
                "diffusion must be an instance of ImutableScalarFunction"
            )
        super().__init__(forcing.basis)
        self._forcing = forcing
        self._diffusion = diffusion
        self._flux_jacobian_implemented = True

    def residual(self, sol: ScalarFunction):
        if not isinstance(sol, ScalarSolution):
            raise ValueError("sol must be an instance of ScalarFunction")
        return div(self._diffusion * nabla(sol)) + self._forcing

    def _flux_jacobian(self, sol_array: Array):
        sol = self.separate_solutions(sol_array)
        flux_jac = (self._diffusion * nabla(sol)).get_matrix_jacobian()
        return flux_jac[:, 0, :, :]


class LinearReactionDiffusionEquation(LinearDiffusionEquation):
    def __init__(
        self,
        forcing: ImutableScalarFunction,
        diffusion: ImutableScalarFunction,
        reaction: ImutableScalarFunction,
    ):
        super().__init__(forcing, diffusion)
        if not isinstance(reaction, ImutableScalarFunction):
            raise ValueError(
                "reaction must be an instance of ImutableScalarFunction"
            )
        self._reaction = reaction

    def residual(self, sol: ScalarFunction):
        diff_res = super().residual(sol)
        react_res = self._reaction * sol
        return diff_res + react_res


class LinearReactionAdvectionDiffusionEquation(LinearDiffusionEquation):
    def __init__(
        self,
        forcing: ImutableScalarFunction,
        diffusion: ImutableScalarFunction,
        reaction: ImutableScalarFunction,
        velocities: ImutableVectorFunction,
    ):
        super().__init__(forcing, diffusion)
        if not isinstance(velocities, ImutableVectorFunction):
            raise ValueError(
                "velocities must be an instance of ImutableVectorFunction"
            )
        self._velocities = velocities

    def residual(self, sol: ScalarFunction):
        react_diff_res = super().residual(sol)
        advec_res = sol * self._velocities
        return react_diff_res + advec_res
