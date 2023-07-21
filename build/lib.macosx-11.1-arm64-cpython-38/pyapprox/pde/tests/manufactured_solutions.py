import numpy as np
import sympy as sp
from functools import partial

def evaluate_sp_lambda(sp_lambda, xx, sample):
    # sp_lambda returns a single function output
    sp_args = tuple(x for x in xx)+tuple(z for z in sample)
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


def evaluate_list_of_sp_lambda(sp_lambdas, xx, sample, as_list=False):
    # sp_lambda returns list of values from multiple functions
    vals = [evaluate_sp_lambda(sp_lambda, xx, sample)
            for sp_lambda in sp_lambdas]
    if as_list:
        return vals
    return np.hstack(vals)


def setup_steady_advection_diffusion_reaction_manufactured_solution(
        sol_string, diff_string, vel_strings, rate_string):
    nphys_vars = len(vel_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(symbs, sol_expr, "numpy")
    sol_fun = partial(evaluate_sp_lambda, sol_lambda)

    diff_expr = sp.sympify(diff_string)
    diff_lambda = sp.lambdify(symbs, diff_expr, "numpy")
    diff_fun = partial(evaluate_sp_lambda, diff_lambda)
    diffusion_expr = sum([(diff_expr*sol_expr.diff(symb, 1)).diff(symb, 1)
                          for symb in symbs])

    vel_exprs = [sp.sympify(vel_string) for vel_string in vel_strings]
    vel_lambdas = [
        sp.lambdify(symbs, vel_expr, "numpy") for vel_expr in vel_exprs]
    vel_fun = partial(evaluate_list_of_sp_lambda, vel_lambdas, as_list=False)
    advection_expr = sum(
        [vel_expr*sol_expr.diff(symb, 1)
         for vel_expr, symb in zip(vel_exprs, symbs)])

    rate_expr = sp.sympify(rate_string)
    rate_lambda = sp.lambdify(symbs, rate_expr, "numpy")
    rate_fun = partial(evaluate_sp_lambda, rate_lambda)
    reaction_expr = rate_expr*sol_expr**2

    forc_expr = -(diffusion_expr-advection_expr-reaction_expr)
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(evaluate_sp_lambda, forc_lambda)

    flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    flux_funs = partial(evaluate_list_of_sp_lambda, flux_lambdas)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)

    return sol_fun, diff_fun, vel_fun, forc_fun, rate_fun, flux_funs


def setup_steady_advection_diffusion_manufactured_solution(
        sol_string, diff_string, vel_strings):
    sol_fun, diff_fun, vel_fun, forc_fun, rate_fun, flux_funs = (
        setup_steady_advection_diffusion_reaction_manufactured_solution(
            sol_string, diff_string, vel_strings, "0"))
    return sol_fun, diff_fun, vel_fun, forc_fun, flux_funs


def setup_steady_linear_diffusion_reaction_manufactured_solution(
        sol_string, diff_string, mass_string, nphys_vars):
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]
    sol_expr = sp.sympify(sol_string)
    sol_lambda = sp.lambdify(symbs, sol_expr, "numpy")
    sol_fun = partial(evaluate_sp_lambda, sol_lambda)

    diff_expr = sp.sympify(diff_string)
    diff_lambda = sp.lambdify(symbs, diff_expr, "numpy")
    diff_fun = partial(evaluate_sp_lambda, diff_lambda)
    diffusion_expr = sum([(diff_expr*sol_expr.diff(symb, 1)).diff(symb, 1)
                          for symb in symbs])

    mass_expr = sp.sympify(mass_string)
    mass_lambda = sp.lambdify(symbs, mass_expr, "numpy")
    mass_fun = partial(evaluate_sp_lambda, mass_lambda)
    second_expr = mass_expr*sol_expr

    forc_expr = -diffusion_expr+second_expr
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(evaluate_sp_lambda, forc_lambda)

    flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    flux_funs = partial(evaluate_list_of_sp_lambda, flux_lambdas)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)
    print("flux", flux_exprs)

    return sol_fun, diff_fun, mass_fun, forc_fun, flux_funs


def setup_steady_stokes_manufactured_solution(
        velocity_strings, pressure_string):
    nphys_vars = len(velocity_strings)
    sp_x, sp_y = sp.symbols(['x', 'y'])
    symbs = (sp_x, sp_y)[:nphys_vars]

    pres_expr = sp.sympify(pressure_string)
    pres_lambda = sp.lambdify(symbs, pres_expr, "numpy")
    pres_fun = partial(evaluate_sp_lambda, pres_lambda)

    vel_expr = [sp.sympify(s) for s in velocity_strings]
    vel_lambda = [sp.lambdify(symbs, vel, "numpy") for vel in vel_expr]
    vel_fun = partial(evaluate_list_of_sp_lambda, vel_lambda, as_list=True)

    forc_expr = []
    for vel, s1 in zip(vel_expr, symbs):
        forc_expr.append(
            sum([-vel.diff(s2, 2) for s2 in symbs]) +
            pres_expr.diff(s1, 1))
    forc_expr.append(sum([vel.diff(s, 1) for vel, s in zip(vel_expr, symbs)]))
    forc_lambda = [sp.lambdify(symbs, f, "numpy") for f in forc_expr]
    forc_fun = partial(evaluate_list_of_sp_lambda, forc_lambda, as_list=True)

    pres_grad_expr = [pres_expr.diff(s, 1) for s in symbs]
    pres_grad_lambda = [sp.lambdify(symbs, pg, "numpy") for pg in pres_grad_expr]
    pres_grad_fun = partial(
        evaluate_list_of_sp_lambda, pres_grad_lambda, as_list=True)

    return vel_fun, pres_fun, forc_fun, pres_grad_fun
