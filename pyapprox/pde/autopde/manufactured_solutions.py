from functools import partial
import numpy as np
import sympy as sp


def _evaluate_sp_lambda(sp_lambda, xx):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals, dtype=np.double)


def _evaluate_transient_sp_lambda(sp_lambda, xx, time):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args, time)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals, dtype=np.double)


def _evaluate_list_of_sp_lambda(sp_lambdas, xx, as_list=False):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_sp_lambda(sp_lambda, xx)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return np.hstack(vals)


def _evaluate_list_of_transient_sp_lambda(sp_lambdas, xx, time, as_list=False):
    # sp_lambda returns list of values from multiple functions
    vals = [_evaluate_transient_sp_lambda(sp_lambda, xx, time)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return np.hstack(vals)


def setup_advection_diffusion_reaction_manufactured_solution(
        sol_string, diff_string, vel_strings, react_fun, transient=False):
    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    if transient:
        # These manufacture solutions assume
        # only solution and forcing are time dependent
        all_symbs = symbs + (sp.symbols('t'),)
    else:
        all_symbs = symbs
    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(all_symbs, sol_expr, "numpy")
    if transient:
        sol_fun = partial(_evaluate_transient_sp_lambda, sol_lambda)
    else:
        sol_fun = partial(_evaluate_sp_lambda, sol_lambda)

    diff_expr = sp.sympify(diff_string)
    diff_lambda = sp.lambdify(symbs, diff_expr, "numpy")
    diff_fun = partial(_evaluate_sp_lambda, diff_lambda)
    diffusion_expr = sum([(diff_expr*sol_expr.diff(symb, 1)).diff(symb, 1)
                          for symb in symbs])

    vel_exprs = [sp.sympify(vel_string) for vel_string in vel_strings]
    vel_lambdas = [
        sp.lambdify(symbs, vel_expr, "numpy") for vel_expr in vel_exprs]
    vel_fun = partial(_evaluate_list_of_sp_lambda, vel_lambdas, as_list=False)
    advection_expr = sum(
        [vel_expr*sol_expr.diff(symb, 1)
         for vel_expr, symb in zip(vel_exprs, symbs)])

    reaction_expr = react_fun(sol_expr)

    # du/dt - diff + advec + react = forc
    forc_expr = -diffusion_expr+advection_expr+reaction_expr
    if transient:
        forc_expr += sol_expr.diff(all_symbs[-1], 1)

    forc_lambda = sp.lambdify(all_symbs, forc_expr, "numpy")
    if transient:
        forc_fun = partial(_evaluate_transient_sp_lambda, forc_lambda)
    else:
        forc_fun = partial(_evaluate_sp_lambda, forc_lambda)

    # following is true definition of flux
    # flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
    # but in unit tests only grad of sol is considered as flux
    flux_exprs = [sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(all_symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    if transient:
        flux_funs = partial(
            _evaluate_list_of_transient_sp_lambda, flux_lambdas)
    else:
        flux_funs = partial(_evaluate_list_of_sp_lambda, flux_lambdas)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)
    print("vel", vel_exprs)

    return sol_fun, diff_fun, vel_fun, forc_fun, flux_funs


def setup_helmholtz_manufactured_solution(sol_string, wnum_string, nphys_vars):
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]

    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(symbs, sol_expr, "numpy")
    sol_fun = partial(_evaluate_sp_lambda, sol_lambda)

    wnum_expr = sp.sympify(wnum_string)
    wnum_lambda = sp.lambdify(symbs, wnum_expr, "numpy")
    wnum_fun = partial(_evaluate_sp_lambda, wnum_lambda)

    forc_expr = sum([sol_expr.diff(symb, 2) for symb in symbs])+(
        wnum_expr*sol_expr)
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(_evaluate_sp_lambda, forc_lambda)

    flux_exprs = [sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    flux_funs = partial(_evaluate_list_of_sp_lambda, flux_lambdas)

    return sol_fun, wnum_fun, forc_fun, flux_funs


def setup_steady_stokes_manufactured_solution(
        vel_strings, pres_string, navier_stokes=False):
    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]

    pres_expr = sp.sympify(pres_string)
    pres_lambda = sp.lambdify(symbs, pres_expr, "numpy")
    pres_fun = partial(_evaluate_sp_lambda, pres_lambda)

    vel_expr = [sp.sympify(s) for s in vel_strings]
    vel_lambda = [sp.lambdify(symbs, vel, "numpy") for vel in vel_expr]
    vel_fun = partial(_evaluate_list_of_sp_lambda, vel_lambda, as_list=False)

    vel_forc_expr = []
    for vel, s1 in zip(vel_expr, symbs):
        vel_forc_expr.append(
            sum([-vel.diff(s2, 2) for s2 in symbs]) +
            pres_expr.diff(s1, 1))
        if navier_stokes:
            vel_forc_expr[-1] += sum(
                [u*vel.diff(s2, 1) for u, s2 in zip(vel_expr, symbs)])
    vel_forc_lambda = [sp.lambdify(symbs, f, "numpy") for f in vel_forc_expr]
    vel_forc_fun = partial(
        _evaluate_list_of_sp_lambda, vel_forc_lambda, as_list=False)
    pres_forc_expr = sum([vel.diff(s, 1) for vel, s in zip(vel_expr, symbs)])
    pres_forc_lambda = sp.lambdify(symbs, pres_forc_expr, "numpy")
    pres_forc_fun = partial(_evaluate_sp_lambda, pres_forc_lambda)

    vel_grad_funs = []
    for v in vel_expr:
        vel_grad_expr = [v.diff(s, 1) for s in symbs]
        vel_grad_lambda = [
            sp.lambdify(symbs, f, "numpy") for f in vel_grad_expr]
        vel_grad_funs.append(partial(
            _evaluate_list_of_sp_lambda, vel_grad_lambda, as_list=False))

    pres_grad_expr = [pres_expr.diff(s, 1) for s in symbs]
    pres_grad_lambda = [
        sp.lambdify(symbs, pg, "numpy") for pg in pres_grad_expr]
    pres_grad_fun = partial(
        _evaluate_list_of_sp_lambda, pres_grad_lambda, as_list=False)

    return (vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, vel_grad_funs,
            pres_grad_fun)


def setup_shallow_water_wave_equations_manufactured_solution(
        vel_strings, depth_string, bed_string, transient=False):
    # Conservative form
    g = 9.81
    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    if transient:
        all_symbs = symbs + (sp.symbols('t'),)
        eval_sp_lambda = _evaluate_transient_sp_lambda
        eval_list_of_sp_lambda = _evaluate_list_of_transient_sp_lambda
    else:
        all_symbs = symbs
        eval_sp_lambda = _evaluate_sp_lambda
        eval_list_of_sp_lambda = _evaluate_list_of_sp_lambda

    vel_expr = [sp.sympify(s) for s in vel_strings]
    vel_lambda = [sp.lambdify(all_symbs, vel, "numpy") for vel in vel_expr]
    vel_fun = partial(
        eval_list_of_sp_lambda, vel_lambda, as_list=False)

    depth_expr = sp.sympify(depth_string)
    depth_lambda = sp.lambdify(all_symbs, depth_expr, "numpy")
    depth_fun = partial(eval_sp_lambda, depth_lambda)

    bed_expr = sp.sympify(bed_string)
    bed_lambda = sp.lambdify(symbs, bed_expr, "numpy")
    bed_fun = partial(_evaluate_sp_lambda, bed_lambda)

    # dh/dt + d(uh)/dx = f
    depth_forc_expr = sum(
        [(u*depth_expr).diff(s, 1) for u, s in zip(vel_expr, symbs)])
    if transient:
        depth_forc_expr += depth_expr.diff(all_symbs[-1], 1)
    depth_forc_lambda = sp.lambdify(all_symbs, depth_forc_expr, "numpy")
    depth_forc_fun = partial(eval_sp_lambda, depth_forc_lambda)

    # du/dt + d(hu**2+gh**2/2) + gh db/dx = f
    vel_forc_expr = [(vel_expr[0]**2*depth_expr+g*depth_expr**2/2).diff(
        symbs[0], 1)]
    if nphys_vars > 1:
        vel_forc_expr[0] += (vel_expr[0]*vel_expr[1]*depth_expr).diff(
            symbs[1], 1)
        vel_forc_expr.append((vel_expr[0]*vel_expr[1]*depth_expr).diff(
            symbs[0], 1)+(vel_expr[1]**2*depth_expr+g*depth_expr**2/2).diff(
                symbs[1], 1))
    vel_forc_expr = [
        e + g*depth_expr*(bed_expr).diff(s, 1)
        for e, s in zip(vel_forc_expr, symbs)]
    print(vel_forc_expr)
    if transient:
        vel_forc_expr = [
            e + (depth_expr*v).diff(all_symbs[-1], 1)
            for e, v in zip(vel_forc_expr, vel_expr)]
        print(vel_forc_expr)
        # assert False
    vel_forc_lambda = [
        sp.lambdify(all_symbs, f, "numpy") for f in vel_forc_expr]
    vel_forc_fun = partial(
        eval_list_of_sp_lambda, vel_forc_lambda, as_list=False)

    print('b', bed_expr)
    print('h', depth_expr)
    print('v', vel_expr)
    print('fh', depth_forc_expr)
    print('fv', vel_forc_expr)

    return depth_fun, vel_fun, depth_forc_fun, vel_forc_fun, bed_fun


def setup_shallow_shelf_manufactured_solution(
        depth_string, vel_strings, bed_string, beta_string, A, rho):

    g, n = 9.81, 3

    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    eval_sp_lambda = _evaluate_sp_lambda
    eval_list_of_sp_lambda = _evaluate_list_of_sp_lambda

    vel_expr = [sp.sympify(s) for s in vel_strings]
    vel_lambda = [sp.lambdify(symbs, vel, "numpy") for vel in vel_expr]
    vel_fun = partial(
        eval_list_of_sp_lambda, vel_lambda, as_list=False)

    depth_expr = sp.sympify(depth_string)
    depth_lambda = sp.lambdify(symbs, depth_expr, "numpy")
    depth_fun = partial(eval_sp_lambda, depth_lambda)

    beta_expr = sp.sympify(beta_string)
    beta_lambda = sp.lambdify(symbs, beta_expr, "numpy")
    beta_fun = partial(eval_sp_lambda, beta_lambda)

    bed_expr = sp.sympify(bed_string)
    bed_lambda = sp.lambdify(symbs, bed_expr, "numpy")
    bed_fun = partial(eval_sp_lambda, bed_lambda)

    vx = [v.diff(s, 1) for s, v in zip(symbs, vel_expr)]
    if nphys_vars == 1:
        De = (vx[0]**2)**(1/2)
    else:
        wx = [v.diff(s, 1) for s, v in zip(symbs[::-1], vel_expr)]
        De = (vx[0]**2+vx[1]**2+vx[0]*vx[1]+1/4*(wx[0]+wx[1])**2)**(1/2)
        print(vx, wx)
    visc_expr = 1/2*A**(-1/n)*De**((1-n)/(n))

    C = 2*visc_expr*depth_expr

    vel_forc_expr = [0 for ii in range(nphys_vars)]
    if nphys_vars == 1:
        vel_forc_expr[0] = -(C*2*vx[0]).diff(symbs[0], 1)
    else:
        vel_forc_expr[0] = -(
            (C*(2*vx[0]+vx[1])).diff(symbs[0], 1) +
            (C*(wx[0]+wx[1])/2).diff(symbs[1], 1))
        vel_forc_expr[1] = -(
            (C*(wx[0]+wx[1])/2).diff(symbs[0], 1) +
            (C*(vx[0]+2*vx[1])).diff(symbs[1], 1))

    for ii in range(nphys_vars):
        vel_forc_expr[ii] += beta_expr*vel_expr[ii]
        vel_forc_expr[ii] += rho*g*depth_expr*(bed_expr+depth_expr).diff(
            symbs[ii], 1)

    vel_forc_lambda = [
        sp.lambdify(symbs, f, "numpy") for f in vel_forc_expr]
    vel_forc_fun = partial(
        eval_list_of_sp_lambda, vel_forc_lambda, as_list=False)

    depth_forc_expr = sum(
        [(v*depth_expr).diff(symb, 1) for v, symb in zip(vel_expr, symbs)])
    depth_forc_fun = partial(
        eval_sp_lambda, sp.lambdify(symbs, depth_forc_expr, "numpy"))

    print('H', depth_expr)
    print('v', vel_expr)
    print('F', vel_forc_expr)
    print('bed', bed_expr)
    print('beta', beta_expr)

    return depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun, depth_forc_fun


def setup_first_order_stokes_ice_manufactured_solution(
        depth_string, vel_strings, bed_string, beta_string, A, rho, g,
        alpha, n, L, return_expr=False):

    # There is no equation for z velocity but mesh must always include z
    nphys_vars = len(vel_strings)+1
    assert nphys_vars == 2
    sp_x, sp_z = sp.symbols(['x', 'z'])
    symbs = (sp_x, sp_z)
    eval_sp_lambda = _evaluate_sp_lambda
    eval_list_of_sp_lambda = _evaluate_list_of_sp_lambda

    vel_expr = [sp.sympify(s) for s in vel_strings]
    vel_lambda = [sp.lambdify(symbs, vel, "numpy") for vel in vel_expr]
    vel_fun = partial(
        eval_list_of_sp_lambda, vel_lambda, as_list=False)

    depth_expr = sp.sympify(depth_string)
    depth_lambda = sp.lambdify(symbs[0], depth_expr, "numpy")
    depth_fun = partial(eval_sp_lambda, depth_lambda)

    beta_expr = sp.sympify(beta_string)
    beta_lambda = sp.lambdify(symbs[0], beta_expr, "numpy")
    beta_fun = partial(eval_sp_lambda, beta_lambda)

    bed_expr = sp.sympify(bed_string)
    bed_lambda = sp.lambdify(symbs[0], bed_expr, "numpy")
    bed_fun = partial(eval_sp_lambda, bed_lambda)

    ux = [vel_expr[0].diff(s, 1) for s in symbs]
    De = (ux[0]**2+ux[1]**2/4)**(1/2)
    visc_expr = 1/2*A**(-1/n)*De**((1-n)/(n))
    # print(visc_expr, 1/2*A**(-1/n))
    C = 2*visc_expr
    vel_forc_expr = [
        -((C*2*ux[0]).diff(symbs[0], 1)+(C*ux[1]/2).diff(symbs[1], 1))]
    # vel_forc_expr[0] -= rho*g*(depth_expr+bed_expr).diff(symbs[0])
    vel_forc_lambda = [
        sp.lambdify(symbs, f, "numpy") for f in vel_forc_expr]
    vel_forc_fun = partial(
        eval_list_of_sp_lambda, vel_forc_lambda, as_list=False)

    # Let w = velocity, u,v be user coordates, x, y canonical coords
    # 2*C*dw/du = 2*C*(dw/dx*dx/du+dw/dy*dy/du)
    #           = 2*C()

    surface_expr = bed_expr+depth_expr
    surface_normal = [-surface_expr.diff(sp_x, 1), 1]
    factor = sum([s**2 for s in surface_normal])**(1/2)
    surface_normal = [s/factor for s in surface_normal]
    # print('n', surface_normal)
    bndry_expr = [
        -C*2*ux[0], C*2*ux[0],
        -C*2*ux[0]*surface_normal[0]-C*ux[1]/2*surface_normal[1]+beta_expr*vel_expr[0],
        C*2*ux[0]*surface_normal[0]+C*ux[1]/2*surface_normal[1],
    ]
    bndry_lambdas = [
        sp.lambdify(symbs, f, "numpy") for f in bndry_expr]
    bndry_funs = [partial(eval_sp_lambda, lam) for lam in bndry_lambdas]

    # import matplotlib.pyplot as plt
    # xx = np.linspace(-L, L, 101)[None, :]
    # zz = (bed_fun(xx)+depth_fun(xx)).T*0+1
    # pts = np.vstack((xx, zz))
    # print(pts.shape)
    # vals = vel_fun(np.vstack((xx, zz)))
    # vals = partial(eval_sp_lambda, sp.lambdify(symbs, C*2*ux[0], "numpy"))(pts)
    # vals = partial(eval_sp_lambda, sp.lambdify(symbs, visc_expr, "numpy"))(pts)
    # vals = partial(eval_sp_lambda, sp.lambdify(symbs, De, "numpy"))(pts)
    # vals = partial(eval_sp_lambda, sp.lambdify(symbs, ux[0], "numpy"))(pts)
    # vals = partial(eval_sp_lambda, sp.lambdify(symbs, vel_expr[0], "numpy"))(pts)
    # plt.plot(xx[0, :], vals, '-')
    # from pyapprox.util.visualization import get_meshgrid_function_data
    # #X, Y, Z = get_meshgrid_function_data(vel_forc_fun, [-1, 1, 0, 1], 50)
    # #plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 20))
    # plt.figure()
    # def fun(x):
    #     x[1] += bed_fun(x[0:1])[:, 0]
    #     print(bed_fun(x[0:1])[:, 0])
    #     return vel_fun(x)
    # X, Y, Z = get_meshgrid_function_data(vel_fun, [-50, 50, 0, 1], 50)
    # p = plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 30))
    # plt.colorbar(p)

    # print(vel_expr)
    # print(vel_forc_expr)

    if return_expr:
        return (depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun,
                bndry_funs, depth_expr, vel_expr, vel_forc_expr, bed_expr,
                beta_expr, bndry_expr, ux, visc_expr, surface_normal)
    return depth_fun, vel_fun, vel_forc_fun, bed_fun, beta_fun, bndry_funs


from pyapprox.pde.autopde.mesh import (
    vertical_transform_2D_mesh, vertical_transform_2D_mesh_inv,
    vertical_transform_2D_mesh_inv_dxdu,
    vertical_transform_2D_mesh_inv_dxdv,
    vertical_transform_2D_mesh_inv_dydu,
    vertical_transform_2D_mesh_inv_dydv
)
from pyapprox.pde.autopde.solvers import Function
def get_vertical_2d_mesh_transforms_from_string(
        xdomain_bounds, surface_string, bed_string):
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)
    bed_expr = sp.sympify(bed_string)
    bed_fun = partial(
        _evaluate_sp_lambda, sp.lambdify(symbs[0], bed_expr, "numpy"))

    surface_expr = sp.sympify(surface_string)
    surface_fun = partial(
        _evaluate_sp_lambda, sp.lambdify(symbs[0], surface_expr, "numpy"))

    surf_grad_u_expr = surface_expr.diff(symbs[0], 1)
    surf_grad_u = partial(
        _evaluate_sp_lambda, sp.lambdify(symbs[0], surf_grad_u_expr, "numpy"))
    bed_grad_u_expr = bed_expr.diff(symbs[0], 1)
    bed_grad_u = partial(
        _evaluate_sp_lambda, sp.lambdify(symbs[0], bed_grad_u_expr, "numpy"))

    print(surface_expr)
    print(bed_expr)

    transform = partial(
        vertical_transform_2D_mesh, xdomain_bounds, bed_fun, surface_fun)
    transform_inv = partial(
        vertical_transform_2D_mesh_inv, xdomain_bounds, bed_fun, surface_fun)
    transform_inv_dxdu = partial(
        vertical_transform_2D_mesh_inv_dxdu, xdomain_bounds)
    transform_inv_dydu = partial(
        vertical_transform_2D_mesh_inv_dydu, bed_fun,
        surface_fun, bed_grad_u, surf_grad_u)
    transform_inv_dxdv = vertical_transform_2D_mesh_inv_dxdv
    transform_inv_dydv = partial(
        vertical_transform_2D_mesh_inv_dydv, bed_fun,
        surface_fun)

    bndry_normals = []
    bndry_normals.append(Function(partial(
        _evaluate_list_of_sp_lambda,
        [sp.lambdify(symbs, sp.sympify("-1"), "numpy"),
         sp.lambdify(symbs, sp.sympify("0"), "numpy")])))
    bndry_normals.append(Function(partial(
        _evaluate_list_of_sp_lambda,
        [sp.lambdify(symbs, sp.sympify("1"), "numpy"),
         sp.lambdify(symbs, sp.sympify("0"), "numpy")])))
    factor = (bed_grad_u_expr**2+1**2)**(1/2)
    bndry_normals.append(Function(partial(
        _evaluate_list_of_sp_lambda,
        [sp.lambdify(symbs, bed_grad_u_expr/factor, "numpy"),
        sp.lambdify(symbs, -sp.sympify("1")/factor, "numpy")])))
    bndry_normals.append(Function(partial(
        _evaluate_list_of_sp_lambda,
        [sp.lambdify(symbs, -surf_grad_u_expr/factor, "numpy"),
         sp.lambdify(symbs, sp.sympify("1")/factor, "numpy")])))

    return (transform, transform_inv,
            [[Function(transform_inv_dxdu, oned=True),
              Function(transform_inv_dydu, oned=True)],
             [Function(transform_inv_dxdv, oned=True),
              Function(transform_inv_dydv, oned=True)]],
            bndry_normals)


def setup_shallow_ice_manufactured_solution(
        depth_string, bed_string, beta_string, A, rho, n, g,
        nphys_vars, transient):
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    if transient:
        all_symbs = symbs + (sp.symbols('t'),)
        eval_sp_lambda = _evaluate_transient_sp_lambda
        eval_list_of_sp_lambda = _evaluate_list_of_transient_sp_lambda
    else:
        all_symbs = symbs
        eval_sp_lambda = _evaluate_sp_lambda
        eval_list_of_sp_lambda = _evaluate_list_of_sp_lambda

    bed_expr = sp.sympify(bed_string)
    bed_lambda = sp.lambdify(symbs, bed_expr, "numpy")
    bed_fun = partial(_evaluate_sp_lambda, bed_lambda)

    depth_expr = sp.sympify(depth_string)
    depth_lambda = sp.lambdify(all_symbs, depth_expr, "numpy")
    depth_fun = partial(eval_sp_lambda, depth_lambda)

    beta_expr = sp.sympify(beta_string)
    beta_lambda = sp.lambdify(symbs, beta_expr, "numpy")
    beta_fun = partial(eval_sp_lambda, beta_lambda)

    surface_expr = bed_expr+depth_expr
    surface_grad_exprs = [surface_expr.diff(s, 1) for s in symbs]

    gamma = 2*A*(rho*g)**n/(n+2)
    forc_expr = -sum([
        ((gamma*depth_expr**(n+2)*(gs**2)**((n-1)/2)+rho*g/beta_expr*depth_expr**2)*gs).diff(s, 1)
        for s, gs in zip(symbs, surface_grad_exprs)])
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(_evaluate_sp_lambda, forc_lambda)

    # dh/dt = N(u)+f
    # f = dh/dt-N(u)

    if transient:
        forc_expr += depth_expr.diff(all_symbs[-1], 1)

    flux_exprs = [depth_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(all_symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    if transient:
        flux_funs = partial(
            _evaluate_list_of_transient_sp_lambda, flux_lambdas)
    else:
        flux_funs = partial(_evaluate_list_of_sp_lambda, flux_lambdas)

    print(depth_expr)
    print(bed_expr)
    print(forc_expr)

    return depth_fun, bed_fun, beta_fun, forc_fun, flux_funs
