import unittest
import numpy as np
import sympy as sp
from functools import partial
import matplotlib.pyplot as plt

from pyapprox.pde.spectral_galerkin import (
    CartesianProductSpectralGalerkinDomain,
    SpectralGalerkinAdvectionDiffusionSolver,
    SpectralGalerkinLinearDiffusionReactionSolver
)
from pyapprox.util.utilities import cartesian_product

def evaluate_sp_lambda(sp_lambda, xx, *args):
    # sp_lambda returns a single function output
    sp_args = tuple([x for x in xx])+args
    vals = sp_lambda(*sp_args)
    if type(vals) == np.ndarray:
        return vals[:, None]
    return np.full((xx.shape[1], 1), vals)


def evaluate_list_of_sp_lambda(sp_lambdas, xx, *args):
    # sp_lambda returns list of values from multiple functions
    vals = [evaluate_sp_lambda(sp_lambda, xx) for sp_lambda in sp_lambdas]
    return np.hstack(vals)


def setup_steady_advection_diffusion_manufactured_solution(
        sol_string, diff_string, vel_strings):
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
    vel_fun = partial(evaluate_list_of_sp_lambda, vel_lambdas)
    advection_expr = sum(
        [vel_expr*sol_expr.diff(symb, 1)
         for vel_expr, symb in zip(vel_exprs, symbs)])

    forc_expr = -(diffusion_expr-advection_expr)
    forc_lambda = sp.lambdify(symbs, forc_expr, "numpy")
    forc_fun = partial(evaluate_sp_lambda, forc_lambda)

    flux_exprs = [diff_expr*sol_expr.diff(symb, 1) for symb in symbs]
    flux_lambdas = [
        sp.lambdify(symbs, flux_expr, "numpy") for flux_expr in flux_exprs]
    flux_funs = partial(evaluate_list_of_sp_lambda, flux_lambdas)

    print("solu", sol_expr)
    print("diff", diff_expr)
    print("forc", forc_expr)

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


class TestSpectralGalerkin(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def check_advection_diffusion(self, domain_bounds, orders, sol_string,
                                  diff_string, vel_strings, bndry_types):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_manufactured_solution(
                sol_string, diff_string, vel_strings))

        def normal_flux(flux_funs, active_var, sign, xx):
            return sign*flux_funs(xx)[:, active_var:active_var+1]

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(nphys_vars):
            if bndry_types[2*dd] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, -1), "N"])
            else:
                bndry_conds.append([lambda xx: sol_fun(xx), "D"])
            if bndry_types[2*dd+1] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, 1), "N"])
            else:
                bndry_conds.append([lambda xx: sol_fun(xx), "D"])

        dirichlet_penalty = 1e7
        domain = CartesianProductSpectralGalerkinDomain(domain_bounds, orders)
        model = SpectralGalerkinAdvectionDiffusionSolver(
            domain, dirichlet_penalty)
        model.initialize(diff_fun, vel_fun, forc_fun, bndry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)

        fig, axs = plt.subplots(1, model.domain._nphys_vars)
        axs = np.atleast_1d(axs)
        p1 = model.domain.plot(sol_fun, 50, ax=axs[0])
        if model.domain._nphys_vars == 1:
            p2 = model.domain.plot_poly(sol, 50, ax=axs[-1], ls='--')
        else:
            p2 = model.domain.plot(
                lambda xx: model.domain.interpolate(sol, xx)-sol_fun(xx),
                50, ax=axs[-1])
            plt.colorbar(p1, ax=axs[0])
            plt.colorbar(p2, ax=axs[-1])
        plt.show()

        xx = cartesian_product(
            [np.linspace(domain_bounds[2*ii], domain_bounds[2*ii+1], 20)
             for ii in range(model.domain._nphys_vars)])
        sol_vals = model.domain.interpolate(sol, xx)
        exact_sol_vals = sol_fun(xx)
        print(sol_vals[[0, -1]])
        print(exact_sol_vals[[0, -1]])
        # print(sol_vals-exact_sol_vals)
        assert np.allclose(sol_vals, exact_sol_vals)


    def test_advection_diffusion(self):
        test_cases = [
            [[0, 1], [3], "x**2", "1", ["2"], ["D"]*2],
            [[0, 1, 0, 1], [2, 2], "x**2*y**2", "1+x", ["0", "0"], ["D"]*4],
            [[0, 1, 0, 1], [2, 2], "x**2*y**2", "1+x", ["2-x", "2"], ["D"]*4]
        ]
        for test_case in test_cases:
            self.check_advection_diffusion(*test_case)

    def check_linear_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string,
            mass_string, bndry_types):

        sol_fun, diff_fun, mass_fun, forc_fun, flux_funs = (
            setup_steady_linear_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, mass_string, len(orders)))
        # Solution designed to have zero flux at boundaries.
        # Thus we are isolating testing of assembling bilinear form and rhs

        def normal_flux(flux_funs, active_var, sign, xx):
            return sign*flux_funs(xx)[:, active_var:active_var+1]

        # effects accuracy of solution
        # larger values will enforce dirichlet boundary condition more
        # accurately but will increase condition number of the discretized
        # bilinear form
        dirichlet_penalty = 1e7

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(nphys_vars):
            if bndry_types[2*dd] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, -1), "N"])
            else:
                bndry_conds.append([lambda xx: sol_fun(xx), "D"])
            if bndry_types[2*dd+1] == "N":
                bndry_conds.append(
                    [partial(normal_flux, flux_funs, dd, 1), "N"])
            else:
                bndry_conds.append([lambda xx: sol_fun(xx), "D"])

        domain = CartesianProductSpectralGalerkinDomain(domain_bounds, orders)
        model = SpectralGalerkinLinearDiffusionReactionSolver(
            domain, dirichlet_penalty)
        model.initialize(diff_fun, mass_fun, forc_fun, bndry_conds)

        sample = np.zeros((0, 1))  # dummy
        sol = model.solve(sample)

        # fig, axs = plt.subplots(1, model.domain._nphys_vars)
        # axs = np.atleast_1d(axs)
        # p1 = model.domain.plot(sol_fun, 50, ax=axs[0])
        # if nphys_vars == 1:
        #     p2 = model.domain.plot_poly(sol, 50, ax=axs[-1], ls='--')
        # else:
        #     p2  = model.domain.plot(
        #         lambda xx: model.domain.interpolate(sol, xx)-sol_fun(xx),
        #         50, ax=axs[-1])
        #     plt.colorbar(p1, ax=axs[0])
        #     plt.colorbar(p2, ax=axs[-1])
        # plt.show()

        xx = cartesian_product(
            [np.linspace(domain_bounds[2*ii], domain_bounds[2*ii+1], 20)
             for ii in range(model.domain._nphys_vars)])
        sol_vals = model.domain.interpolate(sol, xx)
        exact_sol_vals = sol_fun(xx)
        # print(sol_vals)
        # print(exact_sol_vals)
        # print(sol_vals-exact_sol_vals)
        assert np.allclose(sol_vals, exact_sol_vals)

    def test_linear_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [3], "5/3*x**2-10/9*x**3", "1", "1", ["N"]*2],
            [[0, 1], [4], "x**3", "1", "1", ["N"]*2],
            [[0, 1, 0, 1], [4, 3], "(5/3*x**2-10/9*x**3)*(5/3*y**2-10/9*y**3)",
             "1", "1", ["N"]*4],
            [[0, 1, 0, 1], [3, 2], "x**2*y**2", "1", "1", ["D"]*4],
            [[0, 1, 0, 1], [3, 2], "x**2*y**2", "1", "1", ["D"]*2+["N"]*2]]

        for ii, test_case in enumerate(test_cases):
            print(ii)
            self.check_linear_diffusion_reaction(
                *test_case)


if __name__ == "__main__":
    stokes_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestSpectralGalerkin)
    unittest.TextTestRunner(verbosity=2).run(stokes_test_suite)
