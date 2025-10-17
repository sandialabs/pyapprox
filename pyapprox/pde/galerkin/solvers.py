from abc import ABC, abstractmethod
import textwrap

import numpy as np
from skfem import condense, asm, LinearForm, Functional

from pyapprox.pde.galerkin.util import forcing_linearform
from pyapprox.pde.galerkin.functions import FEMFunctionTransientMixin


def newton_solve(
    assemble,
    u_init,
    maxiters=10,
    atol=1e-5,
    rtol=1e-5,
    verbosity=0,
    hard_exit=True,
):
    u = u_init.copy()
    it = 0
    while True:
        u_prev = u.copy()
        bilinear_mat, res, D_vals, D_dofs = assemble(u_prev)
        # minus sign because res = -a(u_prev, v) + L(v)
        # todo remove minus sign and just change sign of update u = u + du
        jac = -bilinear_mat
        II = np.setdiff1d(np.arange(jac.shape[0]), D_dofs)
        # compute residual when boundary conditions have been applied
        # This is done by condense so mimic here
        # order of concatenation will be different to in jac and res
        # but this does not matter when computing norm
        if res.ndim != 1:
            msg = "residual the wrong shape"
            raise RuntimeError(msg)
        res_norm = np.linalg.norm(np.concatenate((res[II], D_vals[D_dofs])))
        if it == 0:
            init_res_norm = res_norm
        if verbosity > 1:
            print("Iter", it, "rnorm", res_norm)
        if not np.isfinite(res_norm):
            msg = "Newton solve residual was not finite"
            if hard_exit:
                raise RuntimeError("Newton solve did not converge\n\t" + msg)
            break
        if it > 0 and res_norm < init_res_norm * rtol + atol:
            msg = f"Newton solve: tolerance {atol}+norm(res_init)*{rtol}"
            msg += f" = {init_res_norm*rtol+atol} reached"
            break
        if it > maxiters:
            msg = f"Newton solve maxiters {maxiters} reached"
            if hard_exit:
                raise RuntimeError("Newton solve did not converge\n\t" + msg)
            break
        # newton solve is du = -inv(j)*res u = u + du
        # move minus sign so that du = inv(j)*res u = u - du
        # np.set_printoptions(linewidth=1000)
        # print(res)
        # print(jac.todense())
        du = skfem.solve(*condense(jac, res, x=D_vals, D=D_dofs))
        # print(du)
        # print(du)
        u = u_prev - du
        it += 1

    if verbosity > 0:
        print(msg)
    return u


from pyapprox.util.newton import NewtonSolver, NewtonResidual
from pyapprox.pde.galerkin.physics import Physics, FEMScalarFunction
from pyapprox.util.backends.template import Array
from pyapprox.util.backends.numpy import NumpyMixin
import skfem


class SteadyPhysicsNewtonResidual(NewtonResidual):
    def __init__(self, physics: Physics):
        super().__init__(NumpyMixin)
        self._physics = physics

    def linsolve(self, sol_array: Array, res_array: Array):
        # assumes __call__ called first
        vec = skfem.solve(
            *condense(self._jac, self._res, x=self._D_vals, D=self._D_dofs)
        )
        return vec

    def __call__(self, sol_array: Array):
        bilinear_mat, self._res, self._D_vals, self._D_dofs = (
            self._physics.assemble(sol_array)
        )
        # minus sign because res = -a(u_prev, v) + L(v) = -bilinear_mat + lvec
        # lvec is not returned to this scope
        self._jac = -bilinear_mat
        # compute residual when boundary conditions have been applied
        # This is done by condense in linsolve. But newton requires full
        # residual to compute norm. So create full residual.
        res = self._bkd.copy(self._res)
        # if self._physics._bndry_conds.ndirichlet_boundaries() > 0:
        #     res[
        #         self._physics._basis.get_dofs(
        #             self._physics._bndry_conds._dbndry_names
        #         ).flatten()
        #     ] = 0.0
        # Note the order of concatenation used here will likely be different
        # to in jac and res but this does not matter because newton solve
        # only uses residual to compute norm. residual is passed back to
        # linsolve but we can replace it with self._res at that point
        II = np.setdiff1d(np.arange(self._jac.shape[0]), self._D_dofs)
        res = np.concatenate((self._res[II], self._D_vals[self._D_dofs]))
        return res

    def jacobian(self, sol_array: Array):
        # TODO remove jacobian as required function and remove
        # default implementation of linsolve that calles self.jacobain
        raise NotImplementedError("Should not be called")


class PDESolver(ABC):
    def __init__(self, physics: Physics):
        if not isinstance(physics, Physics):
            raise ValueError("physics must be an instance of Physics")
        self._physics = physics

    @abstractmethod
    def solve(self):
        raise NotImplementedError

    def __repr__(self):
        return "{0}({1}\n)".format(
            self.__class__.__name__,
            textwrap.indent(
                f"\nphysics={self._physics},\nnetwon="
                + str(self.newton_solver),
                prefix="    ",
            ),
        )


class SteadyStatePDE(PDESolver):
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
            SteadyPhysicsNewtonResidual(self._physics)
        )

    def solve(self, init_sol_array: Array) -> Array:
        if init_sol_array is None:
            init_sol_array = self._physics.init_guess()
        return self.newton_solver.solve(init_sol_array)

    def _integrate(self, w) -> Array:
        return w.y

    def integrate(self, values: Array) -> float:
        return Functional(self._integrate).assemble(
            self._physics._basis, y=values
        )

    def L2_error(self, exact_sol: Array, fem_sol: Array):
        error = np.sqrt(self.integrate((exact_sol - fem_sol) ** 2))
        return error


class TransientPDE:
    def __init__(self, physics: Physics, deltat: float, tableau_name: str):
        self._physics = physics
        self._deltat = deltat
        if tableau_name != "im_beuler1":
            raise NotImplementedError(f"{tableau_name} not implemented")

        self._newton_kwargs = None
        self._mass_mat = None
        self._residual_time = None
        self._residual_deltat = None
        self._residual_sol = None

    def _set_physics_time(self, time):
        for fun in self._physics._funs:
            if isinstance(fun, FEMFunctionTransientMixin):
                fun.set_time(time)
        self._physics._bndry_conds.set_time(time)

    def _rhs(self, sol, time):
        self._set_physics_time(time)
        bilinear_mat, linear_vec = self._physics.raw_assemble(sol)
        return linear_vec, -bilinear_mat

    def _backward_euler_residual(self, sol, time, deltat, stage_unknowns):
        active_stage_time = time + deltat
        srhs, jac = self._rhs(stage_unknowns, active_stage_time)
        temp1 = asm(
            LinearForm(forcing_linearform), self._physics._basis, forc=sol
        )
        temp2 = asm(
            LinearForm(forcing_linearform),
            self._physics._basis,
            forc=stage_unknowns,
        )
        residual = srhs * deltat + temp1 - temp2
        return residual, self._mass_mat - deltat * jac

    def _residual_fun(self, stage_unknowns):
        residual, jac = self._backward_euler_residual(
            self._residual_sol,
            self._residual_time,
            self._residual_deltat,
            stage_unknowns,
        )
        jac, residual, D_vals, D_dofs = (
            self._physics.apply_boundary_conditions(
                stage_unknowns, jac, residual
            )
        )
        return jac, residual, D_vals, D_dofs

    def _update(self, sol, time, deltat, init_guess):
        self._residual_sol = sol
        self._residual_time = time
        self._residual_deltat = deltat
        stage_sol = newton_solve(
            self._residual_fun, init_guess, **self._newton_kwargs
        )
        return stage_sol

    def solve(
        self, init_sol, init_time, final_time, verbosity=0, newton_kwargs={}
    ):
        self._newton_kwargs = newton_kwargs
        self._mass_mat = self._physics.mass_matrix()
        sols, times = [], []
        time = init_time
        times.append(time)
        sol = init_sol.copy()
        sols.append(init_sol[:, None])
        while time < final_time - 1e-12:
            if verbosity >= 1:
                print("Time", time)
            deltat = min(self._deltat, final_time - time)
            sol = self._update(sol, time, deltat, sol.copy())
            sols.append(sol[:, None])
            time += deltat
            times.append(time)
        if verbosity >= 1:
            print("Time", time)
        sols = np.hstack(sols)
        return sols, times

    def _integrate(self, w) -> Array:
        return w.y

    def integrate(self, values: Array) -> float:
        return Functional(self._integrate).assemble(
            self._physics._basis, y=values
        )

    def L2_error_at_a_single_time(self, exact_sol: Array, fem_sol: Array):
        error = np.sqrt(self.integrate((exact_sol - fem_sol) ** 2))
        return error
