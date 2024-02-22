import unittest
import numpy as np
import sympy as sp
from functools import partial
import matplotlib.pyplot as plt

from pyapprox.pde.galerkin.spectral_galerkin import (
    CartesianProductSpectralGalerkinDomain,
    SpectralGalerkinAdvectionDiffusionSolver,
    SpectralGalerkinLinearDiffusionReactionSolver
)
from pyapprox.util.utilities import cartesian_product
from pyapprox.pde.tests.manufactured_solutions import (
    setup_steady_advection_diffusion_manufactured_solution,
    setup_steady_linear_diffusion_reaction_manufactured_solution
)


class TestSpectralGalerkin(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def check_advection_diffusion(self, domain_bounds, orders, sol_string,
                                  diff_string, vel_strings, bndry_types):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_manufactured_solution(
                sol_string, diff_string, vel_strings))

        sample = np.zeros((0, 1))  # dummy
        # model should not know about randomness and just take functions
        # even when sample is not a dummy variable
        def normal_flux(flux_funs, active_var, sign, xx):
            return sign*flux_funs(xx, sample)[:, active_var:active_var+1]
        diff_fun = partial(diff_fun, sample=sample)
        vel_fun = partial(vel_fun, sample=sample)
        forc_fun = partial(forc_fun, sample=sample)
        sol_fun = partial(sol_fun, sample=sample)

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

        sol = model.solve()

        # fig, axs = plt.subplots(1, model.domain._nphys_vars)
        # axs = np.atleast_1d(axs)
        # p1 = model.domain.plot(sol_fun, 50, ax=axs[0])
        # if model.domain._nphys_vars == 1:
        #     p2 = model.domain.plot_poly(sol, 50, ax=axs[-1], ls='--')
        # else:
        #     p2 = model.domain.plot(
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
        print(sol_vals[[0, -1]])
        print(exact_sol_vals[[0, -1]])
        # print(sol_vals-exact_sol_vals)
        assert np.allclose(sol_vals, exact_sol_vals)

    def test_advection_diffusion(self):
        test_cases = [
            [[0, 1], [3], "x**2", "1", ["2"], ["D"]*2],
            [[0, 1], [3], "x**2", "1", ["2"], ["D", "N"]],
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

        # model should not know about randomness and just take functions
        # even when sample is not a dummy variable
        sample = np.zeros((0, 1))  # dummy
        def normal_flux(flux_funs, active_var, sign, xx):
            return sign*flux_funs(xx, sample)[:, active_var:active_var+1]
        diff_fun = partial(diff_fun, sample=sample)
        mass_fun = partial(mass_fun, sample=sample)
        forc_fun = partial(forc_fun, sample=sample)
        sol_fun = partial(sol_fun, sample=sample)

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

        sol = model.solve()

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


