import numpy as np
from skfem import condense, solve

def newton_solve(assemble, u_init,
                 maxiters=10, atol=1e-5, rtol=1e-5, verbosity=2,
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
