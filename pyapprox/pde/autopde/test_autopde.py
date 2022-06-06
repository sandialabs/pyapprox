import torch
import unittest
import numpy as np
from functools import partial

from pyapprox.pde.autopde.manufactured_solutions import (
    setup_steady_advection_diffusion_reaction_manufactured_solution,
    setup_helmholtz_manufactured_solution
)
from pyapprox.pde.autopde.autopde import (
    CartesianProductCollocationMesh, AdvectionDiffusionReaction,
    Function, EulerBernoulliBeam, Helmholtz, SteadyStateLinearPDE,
    TransientFunction, TransientPDE
)



# Functions and testing only for wrapping Sympy generated manufactured
# solutions
def _normal_flux(flux_funs, active_var, sign, xx):
    vals = sign*flux_funs(xx)[:, active_var:active_var+1]
    return vals


def _robin_bndry_fun(sol_fun, flux_funs, active_var, sign, alpha, xx,
                     time=None):
    if time is not None:
        if hasattr(sol_fun, "set_time"):
            sol_fun.set_time(time)
        if hasattr(flux_funs, "set_time"):
            flux_funs.set_time(time)
    vals = alpha*sol_fun(xx) + _normal_flux(flux_funs, active_var, sign, xx)
    return vals


class TestAutoPDE(unittest.TestCase):
    def setUp(self):
      torch.manual_seed(1)
      np.random.seed(1)

    def check_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string, vel_strings,
            react_fun, bndry_types):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_fun))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(2*nphys_vars):
            if bndry_types[dd] == "D":
                bndry_conds.append([sol_fun, "D"])
            else:
                if bndry_types[dd] == "R":
                    alpha = 1
                else:
                    alpha = 0
                bndry_conds.append(
                    [partial(_robin_bndry_fun, sol_fun, flux_funs, dd//2,
                             (-1)**(dd+1), alpha), "R", alpha])

        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = SteadyStateLinearPDE(AdvectionDiffusionReaction(
            mesh, diff_fun, vel_fun, react_fun, forc_fun))
        sol = solver.solve()

        # import matplotlib.pyplot as plt
        # ax = plt.subplots(1, 1, figsize=(8, 6))[1]
        # solver.mesh.plot(sol, ax=ax, marker='o', ls=None)
        # solver.mesh.plot(
        #     sol_fun(solver.mesh.mesh_pts).numpy(), ax=ax, marker='s',
        #     ls=None)
        # solver.mesh.plot(sol, ax=ax, nplot_pts_1d=100, label="Collocation")
        # solver.mesh.plot(
        #     sol_fun(solver.mesh.mesh_pts).numpy(), ax=ax, nplot_pts_1d=100,
        #     ls="--", label="Exact")
        # plt.legend()
        # plt.show()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9

        
        # normals = solver.mesh._get_bndry_normals(np.arange(nphys_vars*2))
        # if nphys_vars == 2:
        #     assert np.allclose(
        #         normals, np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]))
        # else:
        #     assert np.allclose(normals, np.array([[-1], [1]]))
        # _normal_fluxes = solver.compute_bndry_fluxes(
        #     sol, np.arange(nphys_vars*2))
        # for ii, indices in enumerate(solver.mesh.boundary_indices):
        #     assert np.allclose(
        #         np.array(_normal_fluxes[ii]),
        #         flux_funs(solver.mesh.mesh_pts[:, indices],).dot(
        #             normals[ii]))
    
    def test_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["D", "D"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["N", "D"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: 0*x**2,
             ["R", "D"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"], lambda x: x**2,
             ["D", "D"]],
            [[0, 1, 0, 1], [4, 4], "y**2*x**2", "1", ["0", "0"],
             lambda x: 0*x**2, ["D", "N", "N", "D"]],
            [[0, .5, 0, 1], [14, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             lambda x: 0*x**2, ["D", "N", "N", "D"]],
            [[0, .5, 0, 1], [16, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             lambda x: 0*x**2, ["D", "R", "D", "D"]]
        ]
        for test_case in test_cases:
            self.check_advection_diffusion_reaction(*test_case)

    def test_euler_bernoulli_beam(self):
        # bndry_conds are None because they are imposed in the solver
        # This 4th order equation requires 4 boundary conditions, two at each
        # end of the domain. This cannot be done with usual boundary condition
        # functions and msut be imposed on the residual exactly
        domain_bounds, orders, bndry_conds = [0, 1], [4], [None]*2
        emod_val, smom_val, forcing_val = 1., 1., -2. 
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = SteadyStateLinearPDE(EulerBernoulliBeam(
            mesh, Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), 1)),
            Function(lambda x: np.full((x.shape[1], 1), forcing_val))))
        sol = solver.solve()

        def sol_fun(x):
            length = domain_bounds[1]-domain_bounds[0]
            return (forcing_val*x**2*(6*length**2-4*length*x+x**2)/(
                24*emod_val*smom_val)).T
        assert np.allclose(sol, sol_fun(mesh.mesh_pts))
        # mesh.plot(sol, nplot_pts_1d=100)
        # import matplotlib.pyplot as plt
        # plt.show()

    def check_helmholtz(self, domain_bounds, orders, sol_string, wnum_string,
                        bndry_types):
        sol_fun, wnum_fun, forc_fun, flux_funs = (
            setup_helmholtz_manufactured_solution(
                sol_string, wnum_string, len(domain_bounds)//2))

        wnum_fun = Function(wnum_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(2*nphys_vars):
            if bndry_types[dd] == "D":
                bndry_conds.append([sol_fun, "D"])
            else:
                if bndry_types[dd] == "R":
                    alpha = 1
                else:
                    alpha = 0
                bndry_conds.append(
                    [partial(_robin_bndry_fun, sol_fun, flux_funs, dd//2,
                             (-1)**(dd+1), alpha), "R", alpha])

        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = SteadyStateLinearPDE(Helmholtz(mesh, wnum_fun, forc_fun))
        sol = solver.solve()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9

    def test_helmholtz(self):
        test_cases = [
            [[0, 1], [16], "x**2", "1", ["N", "D"]],
            [[0, .5, 0, 1], [16, 16], "y**2*x**2", "1", ["N", "D", "D", "D"]]]
        for test_case in test_cases:
            self.check_helmholtz(*test_case)

    def check_transient_pde(self, domain_bounds, orders, sol_string,
                            diff_string, vel_strings, react_fun, bndry_types,
                            tableau_name):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_steady_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_fun, True))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = TransientFunction(forc_fun, name='forcing')
        sol_fun = TransientFunction(sol_fun, name='sol')
        flux_funs = TransientFunction(flux_funs, name='flux')

        nphys_vars = len(orders)
        bndry_conds = []
        for dd in range(2*nphys_vars):
            if bndry_types[dd] == "D":
                bndry_conds.append([sol_fun, "D"])
            else:
                if bndry_types[dd] == "R":
                    alpha = 1
                else:
                    alpha = 0
                bndry_conds.append(
                    [TransientFunction(
                        partial(_robin_bndry_fun, sol_fun, flux_funs, dd//2,
                                (-1)**(dd+1), alpha)), "R", alpha])

        deltat = 0.1
        final_time = deltat*5
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders, bndry_conds)
        solver = TransientPDE(
            AdvectionDiffusionReaction(
                mesh, diff_fun, vel_fun, react_fun, forc_fun),
            deltat, tableau_name)
        sol_fun.set_time(0)
        sols, times = solver.solve(sol_fun(mesh.mesh_pts), 0, final_time)

        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = sol_fun(solver.residual.mesh.mesh_pts).numpy()
            model_sol_t = sols[:, ii:ii+1]
            L2_error = np.sqrt(
                solver.residual.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                solver.residual.mesh.integrate(exact_sol_t**2))
            # print(time, L2_error, 1e-8*factor)
            assert L2_error < 1e-8*factor


    def test_transient_pde(self):
        test_cases = [
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["0"],
             lambda x: 0*x**2, ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             lambda x: 1*x**2, ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             lambda x: 1*x**2, ["N", "D"], "im_crank2"],
            [[0, 1, 0, 1], [3, 3], "(x-1)*x*(1+t)**2*y**2", "1", ["1", "1"],
             lambda x: 1*x**2, ["D", "N", "R", "D"], "im_crank2"]
        ]
        for test_case in test_cases:
            self.check_transient_pde(*test_case)
        

if __name__ == "__main__":
    autopde_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestAutoPDE)
    unittest.TextTestRunner(verbosity=2).run(autopde_test_suite)
