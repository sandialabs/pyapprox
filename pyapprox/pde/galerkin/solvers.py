import numpy as np
from functools import partial
from skfem import condense, solve, asm, LinearForm

from pyapprox.pde.galerkin.util import _forcing


def newton_solve(assemble, u_init,
                 maxiters=10, atol=1e-5, rtol=1e-5, verbosity=0,
                 hard_exit=True):
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
        if it > 0 and res_norm < init_res_norm*rtol+atol:
            msg = f"Netwon solve: tolerance {atol}+norm(res_init)*{rtol}"
            msg += f" = {init_res_norm*rtol+atol} reached"
            break
        if it > maxiters:
            msg = f"Newton solve maxiters {maxiters} reached"
            if hard_exit:
                raise RuntimeError("Newton solve did not converge\n\t"+msg)
            break
        # netwon solve is du = -inv(j)*res u = u + du
        # move minus sign so that du = inv(j)*res u = u - du
        du = solve(*condense(jac, res, x=D_vals, D=D_dofs))
        # print(du)
        u = u_prev - du
        it += 1

    if verbosity > 0:
        print(msg)
    return u


class SteadyStatePDE():
    def __init__(self, physics):
        self.physics = physics

    def solve(self, init_guess=None, **newton_kwargs):
        if init_guess is None:
            init_guess = self.physics.init_guess()
        sol = newton_solve(
            self.physics.assemble, init_guess, **newton_kwargs)
        return sol


class TransientFunction():
    def __init__(self, fun, name="fun"):
        self._fun = fun
        self._name = name

    def __call__(self, samples):
        return self._eval(samples)

    def _eval(self, samples):
        if self._time is None:
            raise ValueError("Must call set_time before calling eval")
        return self._partial_fun(samples)[:, 0]

    def set_time(self, time):
        self._time = time
        self._partial_fun = partial(self._fun, time=time)

    def __repr__(self):
        return "{0}(time={1})".format(self._name, self._time)


class TransientPDE():
    def __init__(self, physics, deltat, tableau_name):
        self.physics = physics
        self._deltat = deltat
        if tableau_name != "im_beuler1":
            raise NotImplementedError(f"{tableau_name} not implemented")

        self._newton_kwargs = None
        self._mass_mat = None
        self._residual_time = None
        self._residual_deltat = None
        self._residual_sol = None

    def _set_physics_time(self, time):
        for fun in self.physics.funs:
            if hasattr(fun, "set_time"):
                fun.set_time(time)
        # iterate over dirichlet, neumann and robin BC types
        for bndry_cond in self.physics.bndry_conds:
            # iterate over all BCs of the current type
            for bc_name, bc in bndry_cond.items():
                if hasattr(bc[0], "set_time"):
                    bc[0].set_time(time)

    def _rhs(self, sol, time):
        self._set_physics_time(time)
        bilinear_mat, linear_vec = self.physics.raw_assemble(sol)
        rhs = -bilinear_mat.dot(sol) + linear_vec
        # print(rhs.shape, bilinear_mat.shape)
        return rhs, -bilinear_mat

    def _diag_runge_kutta_stage_solution(
            self, sol, time, deltat, stage_unknowns):
        active_stage_time = time+deltat
        active_stage_sol = stage_unknowns
        srhs, jac = self._rhs(active_stage_sol, active_stage_time)
        temp = asm(LinearForm(_forcing), self.physics.basis, forc=sol)
        new_active_stage_unknowns = (temp+srhs*deltat)
        return new_active_stage_unknowns, srhs, jac

    def _diag_residual_fun(self, stage_unknowns):
        out = self._diag_runge_kutta_stage_solution(
            self._residual_sol, self._residual_time, self._residual_deltat,
            stage_unknowns)
        stage_jac = self._residual_deltat*out[2]
        jac = self._mass_mat-stage_jac
        new_active_stage_unknowns = out[0]
        residual = self._mass_mat.dot(stage_unknowns)-new_active_stage_unknowns
        jac, residual, D_vals, D_dofs = (
            self.physics.apply_dirichlet_boundary_conditions(
                new_active_stage_unknowns, jac, residual))
        return jac, residual, D_vals, D_dofs

    def _update(self, sol, time, deltat, init_guess):
        self._residual_sol = sol
        self._residual_time = time
        self._residual_deltat = deltat
        stage_sol = newton_solve(
            self._diag_residual_fun, init_guess, **self._newton_kwargs)
        return stage_sol

    def solve(self, init_sol, init_time, final_time, verbosity=0,
              newton_kwargs={}):
        self._newton_kwargs = newton_kwargs
        self._mass_mat = self.physics.mass_matrix()
        sols, times = [], []
        time = init_time
        times.append(time)
        sol = init_sol.copy()
        while time < final_time-1e-12:
            if verbosity >= 1:
                print("Time", time)
            deltat = min(self._deltat, final_time-time)
            sol = self._update(
                sol, time, deltat, sol.copy())
            sols.append(sol.detach())
            time += deltat
            times.append(time)
        if verbosity >= 1:
            print("Time", time)
        sols = np.stack(sols, dim=1)
        return sols, times
