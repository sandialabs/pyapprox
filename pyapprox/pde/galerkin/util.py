import numpy as np
from skfem import (
    MeshLine, MeshQuad, Mesh2D, Mesh3D,
    ElementLineP1, ElementLineP2, ElementQuad1, ElementQuad2,
    condense, solve)
from skfem.helpers import dot, grad

def _get_mesh(bounds, nrefine):
    nphys_vars = len(bounds)//2
    if nphys_vars > 2:
        raise ValueError("Only 1D and 2D meshes supported")

    if nphys_vars == 1:
        mesh = (MeshLine.init_tensor(np.linspace(*bounds, 3))
                .refined(nrefine)
                .with_boundaries({
                    "left": lambda x: x[0] == 0,
                    "right": lambda x: x[0] == 1}))
        return mesh

    mesh = (
        MeshQuad.init_tensor(
            np.linspace(*bounds[:2], 3), np.linspace(*bounds[2:], 3))
        .refined(nrefine)
        .with_boundaries({
            "left": lambda x: x[0] == bounds[0],
            "right": lambda x: x[0] == bounds[1],
            "bottom": lambda x: x[1] == bounds[2],
            "top": lambda x: x[1] == bounds[3]}))
    return mesh


def _get_element(mesh, order):

    if order > 2:
        raise ValueError(f"order {order} not supported")
    if isinstance(mesh, Mesh3D):
        raise ValueError("Only 1D and 2D meshes supported")

    nphys_vars = (2 if isinstance(mesh, Mesh2D) else 1)

    if nphys_vars == 1:
        if order == 1:
            return ElementLineP1()
        return ElementLineP2()
    if order == 1:
        return ElementQuad1()
    return ElementQuad2()


def _vector_fun_to_skfem_vector_fun(vel_forc_fun, x):
    vals = vel_forc_fun(x)
    return np.stack([vals[:, ii] for ii in range(vals.shape[1])])


def _vector_forcing(v, w):
    return dot(w["forc"],  v)


def _forcing(v, w):
    return w["forc"] * v


def _robin(u, v, w):
    return w["alpha"] * u * v


def _robin_prev_sol(v, w):
    return w["alpha"] * w["u_prev"] * v



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
