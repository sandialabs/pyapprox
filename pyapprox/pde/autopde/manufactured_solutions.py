from functools import partial
import numpy as np
import sympy as sp


def _evaluate_sp_lambda(sp_lambda, xx):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


def _evaluate_transient_sp_lambda(sp_lambda, xx, time):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)
    vals = sp_lambda(*sp_args, time)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


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

    forc_expr = -diffusion_expr+advection_expr+reaction_expr
    if transient:
        forc_expr += sol_expr.diff(all_symbs[-1], 1)

    forc_lambda = sp.lambdify(all_symbs, forc_expr, "numpy")
    if transient:
        forc_fun = partial(_evaluate_transient_sp_lambda, forc_lambda)
    else:
        forc_fun = partial(_evaluate_sp_lambda, forc_lambda)

    flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
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

    pres_grad_expr = [pres_expr.diff(s, 1) for s in symbs]
    pres_grad_lambda = [
        sp.lambdify(symbs, pg, "numpy") for pg in pres_grad_expr]
    pres_grad_fun = partial(
        _evaluate_list_of_sp_lambda, pres_grad_lambda, as_list=True)

    return vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, pres_grad_fun


def setup_shallow_wave_equations_manufactured_solution(
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

    depth_forc_expr = sum(
        [(u*depth_expr).diff(s, 1) for u, s in zip(vel_expr, symbs)])
    if transient:
        depth_forc_expr += depth_expr.diff(all_symbs[-1], 1)
    depth_forc_lambda = sp.lambdify(all_symbs, depth_forc_expr, "numpy")
    depth_forc_fun = partial(eval_sp_lambda, depth_forc_lambda)

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
    if transient:
        vel_forc_expr = [
            e + (depth_expr*v).diff(all_symbs[-1], 1)
            for e, v in zip(vel_forc_expr, vel_expr)]
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
        De = (vx[0]**2/2)**(1/2)
    else:
        wx = [v.diff(s, 1) for s, v in zip(symbs[::-1], vel_expr)]
        De = (vx[0]**2+vx[1]**2+vx[0]*vx[1]+1/4*(wx[0]+wx[1])**2)**(1/2)
        print(vx, wx)
    visc_expr = 1/2*A**(-1/n)*De**((n-1)/(n))

    C = 2*visc_expr*depth_expr

    forc_expr = [0 for ii in range(nphys_vars)]
    if nphys_vars == 1:
        forc_expr[0] = -(C*2*vx[0]).diff(symbs[0], 1)
    else:
        forc_expr[0] = -(
            (C*(2*vx[0]+vx[1])).diff(symbs[0], 1) +
            (C*(wx[0]+wx[1])/2).diff(symbs[1], 1))
        forc_expr[1] = -(
            (C*(wx[0]+wx[1])/2).diff(symbs[0], 1) +
            (C*(vx[0]+2*vx[1])).diff(symbs[1], 1))

    for ii in range(nphys_vars):
        forc_expr[ii] += beta_expr*vel_expr[ii]
        forc_expr[ii] += rho*g*depth_expr*(bed_expr+depth_expr).diff(
            symbs[ii], 1)

    forc_lambda = [
        sp.lambdify(symbs, f, "numpy") for f in forc_expr]
    forc_fun = partial(
        eval_list_of_sp_lambda, forc_lambda, as_list=False)

    print('H', depth_expr)
    print('v', vel_expr)
    print('F', forc_expr)
    print('bed', bed_expr)
    print('beta', beta_expr)

    return depth_fun, vel_fun, forc_fun, bed_fun, beta_fun
