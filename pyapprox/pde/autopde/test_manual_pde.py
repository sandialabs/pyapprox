import unittest
import torch
import numpy as np
from functools import partial

from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
    get_vertical_2d_mesh_transforms_from_string,
    setup_steady_stokes_manufactured_solution,
    setup_shallow_ice_manufactured_solution
)
from pyapprox.pde.autopde.test_autopde import (
    _get_boundary_funs, _vel_component_fun
)
from pyapprox.pde.autopde.autopde import (
    CartesianProductCollocationMesh,
    TransformedCollocationMesh, InteriorCartesianProductCollocationMesh,
    TransformedInteriorCollocationMesh, VectorMesh,
    Function, TransientFunction, SteadyStatePDE, TransientPDE)
from pyapprox.pde.autopde.manual_pde import (
    AdvectionDiffusionReaction, IncompressibleNavierStokes,
    LinearIncompressibleStokes, ShallowIce
)
from pyapprox.util.utilities import approx_jacobian


class TestManualPDE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def _check_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, basis_types, mesh_transforms=None):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0]))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = Function(forc_fun)
        sol_fun = Function(sol_fun)

        nphys_vars = len(orders)

        if mesh_transforms is None:
            bndry_conds = _get_boundary_funs(
                nphys_vars, bndry_types, sol_fun, flux_funs)
            mesh = CartesianProductCollocationMesh(
                domain_bounds, orders, basis_types)
        else:
            bndry_conds = _get_boundary_funs(
                nphys_vars, bndry_types, sol_fun, flux_funs,
                mesh_transforms[-1])
            mesh = TransformedCollocationMesh(
                orders, *mesh_transforms)

        solver = SteadyStatePDE(AdvectionDiffusionReaction(
            mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
            react_funs[1]))

        assert np.allclose(
            solver.residual._raw_residual(sol_fun(mesh.mesh_pts)[:, 0])[0], 0)
        assert np.allclose(
            solver.residual._residual(sol_fun(mesh.mesh_pts)[:, 0])[0], 0)
        sol = solver.solve()

        print(np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol))
        assert np.linalg.norm(
            sol_fun(mesh.mesh_pts)-sol) < 1e-9

    def test_advection_diffusion_reaction(self):
        s0, depth, L, alpha = 2, .1, 1, 1e-1
        mesh_transforms = get_vertical_2d_mesh_transforms_from_string(
            [-L, L], f"{s0}-{alpha}*x**2-{depth}", f"{s0}-{alpha}*x**2")

        test_cases = [
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
                [lambda sol: 0*sol,
                lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["N", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["R", "D"], ["C"]],
            [[0, 1], [4], "0.5*(x-3)*x", "1", ["0"],
             [lambda sol: sol**2, lambda sol: np.diag(2*sol[:, 0])],
             ["D", "D"], ["C"]],
            # When using periodic bcs must have reaction term to have a
            # unique solution
            [[0, 2*np.pi], [30], "sin(x)", "1", ["0"],
             [lambda sol: 1*sol, lambda sol: np.eye(sol.shape[0])],
             ["P", "P"], ["C"]],
            [[0, 1, 0, 1], [3, 3], "y**2*x**2", "1", ["0", "0"], #[4, 4]
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "N", "N", "D"], ["C", "C"]],
            [[0, .5, 0, 1], [14, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "N", "N", "D"], ["C", "C"]],
            [[0, .5, 0, 1], [16, 16], "y**2*sin(pi*x)", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "R", "D", "D"], ["C", "C"]],
            [None, [6, 6], "y**2*x**2", "1", ["0", "0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "D", "D", "D"], ["C", "C"], mesh_transforms],
            [None, [6, 6], "y**2*x**2", "1", ["1", "0"],
             [lambda sol: 1*sol**2,
              lambda sol: np.diag(2*sol[:, 0])],
             ["D", "D", "D", "N"], ["C", "C"],
             mesh_transforms]
        ]
        for test_case in test_cases:
            self._check_advection_diffusion_reaction(*test_case)


    def _check_transient_advection_diffusion_reaction(
            self, domain_bounds, orders, sol_string,
            diff_string, vel_strings, react_funs, bndry_types,
            tableau_name):
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], True))

        diff_fun = Function(diff_fun)
        vel_fun = Function(vel_fun)
        forc_fun = TransientFunction(forc_fun, name='forcing')
        sol_fun = TransientFunction(sol_fun, name='sol')
        flux_funs = TransientFunction(flux_funs, name='flux')

        nphys_vars = len(orders)
        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, sol_fun, flux_funs)

        deltat = 0.1
        final_time = deltat*1# 5
        mesh = CartesianProductCollocationMesh(domain_bounds, orders)
        solver = TransientPDE(
            AdvectionDiffusionReaction(
                mesh, bndry_conds, diff_fun, vel_fun, react_funs[0], forc_fun,
                react_funs[1]), deltat, tableau_name)
        sol_fun.set_time(0)
        sols, times = solver.solve(
            sol_fun(mesh.mesh_pts), 0, final_time, newton_opts={"tol": 1e-8})

        for ii, time in enumerate(times):
            sol_fun.set_time(time)
            exact_sol_t = sol_fun(solver.residual.mesh.mesh_pts).numpy()
            model_sol_t = sols[:, ii:ii+1]
            print(exact_sol_t)
            print(model_sol_t)
            L2_error = np.sqrt(
                solver.residual.mesh.integrate((exact_sol_t-model_sol_t)**2))
            factor = np.sqrt(
                solver.residual.mesh.integrate(exact_sol_t**2))
            print(time, L2_error, 1e-8*factor)
            assert L2_error < 1e-8*factor

    def test_transient_advection_diffusion_reaction(self):
        test_cases = [
            [[0, 1], [3], "x**2*(1+t)", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "D"], "ex_feuler1"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["0"],
             [lambda sol: 0*sol,
              lambda sol: np.zeros((sol.shape[0], sol.shape[0]))],
             ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             [lambda sol: 1*sol**2,
              lambda sol: np.diag(2*sol[:, 0])],
             ["D", "D"], "im_crank2"],
            [[0, 1], [3], "(x-1)*x*(1+t)**2", "1", ["1"],
             [lambda sol: 1*sol**2,
              lambda sol: np.diag(2*sol[:, 0])],
             ["N", "D"], "im_crank2"],
            [[0, 1, 0, 1], [3, 3], "(x-1)*x*(1+t)**2*y**2", "1", ["1", "1"],
             [lambda sol: 1*sol**2,
              lambda sol: np.diag(2*sol[:, 0])],
             ["D", "N", "R", "D"], "im_crank2"]
        ]
        for test_case in test_cases[:1]:
            self._check_transient_advection_diffusion_reaction(*test_case)

    def _check_stokes_solver_mms(
            self, domain_bounds, orders, vel_strings, pres_string, bndry_types,
            navier_stokes, mesh_transforms=None):
        (vel_fun, pres_fun, vel_forc_fun, pres_forc_fun, vel_grad_funs,
         pres_grad_fun) = setup_steady_stokes_manufactured_solution(
                vel_strings, pres_string, navier_stokes)

        vel_fun = Function(vel_fun)
        pres_fun = Function(pres_fun)
        vel_forc_fun = Function(vel_forc_fun)
        pres_forc_fun = Function(pres_forc_fun)
        pres_grad_fun = Function(pres_grad_fun)

        # TODO Curently not test stokes with Neumann Boundary conditions

        nphys_vars = len(orders)
        if mesh_transforms is None:
            boundary_normals = None
        else:
            boundary_normals = mesh_transforms[-1]
        vel_bndry_conds = [
            _get_boundary_funs(
                nphys_vars, bndry_types,
                partial(_vel_component_fun, vel_fun, ii),
                vel_grad_funs[ii], boundary_normals)
            for ii in range(nphys_vars)]
        bndry_conds = vel_bndry_conds + [[[None, None]]*(2*nphys_vars)]

        if mesh_transforms is None:
            vel_meshes = [
                CartesianProductCollocationMesh(
                    domain_bounds, orders)]*nphys_vars
            pres_mesh = InteriorCartesianProductCollocationMesh(
                domain_bounds, orders)
        else:
            vel_meshes = [TransformedCollocationMesh(
                orders, *mesh_transforms)]*nphys_vars
            pres_mesh = TransformedInteriorCollocationMesh(
                orders, *mesh_transforms[:-1])
        mesh = VectorMesh(vel_meshes + [pres_mesh])
        pres_idx = 0
        pres_val = pres_fun(pres_mesh.mesh_pts[:, pres_idx:pres_idx+1])
        if not navier_stokes:
            Residual = LinearIncompressibleStokes
        else:
            Residual = IncompressibleNavierStokes
        solver = SteadyStatePDE(Residual(
            mesh, bndry_conds, vel_forc_fun, pres_forc_fun,
            (pres_idx, pres_val)))

        exact_vel_vals = vel_fun(vel_meshes[0].mesh_pts).numpy()
        exact_pres_vals = pres_fun(pres_mesh.mesh_pts).numpy()
        exact_sol = torch.vstack(
            [v[:, None] for v in vel_fun(vel_meshes[0].mesh_pts).T] +
            [pres_fun(pres_mesh.mesh_pts)])

        print(np.abs(solver.residual._raw_residual(exact_sol[:, 0])[0]).max())
        assert np.allclose(
            solver.residual._raw_residual(exact_sol[:, 0])[0], 0, atol=2e-8)
        assert np.allclose(
            solver.residual._residual(exact_sol[:, 0])[0], 0, atol=2e-8)

        def fun(s):
            return solver.residual._raw_residual(torch.as_tensor(s))[0].numpy()
        j_fd = approx_jacobian(fun, exact_sol[:, 0].numpy())
        j_man = solver.residual._raw_residual(torch.as_tensor(exact_sol[:, 0]))[1].numpy()
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.residual._raw_residual(s)[0],
            exact_sol[:, 0].clone().requires_grad_(True), strict=True).numpy()
        np.set_printoptions(precision=2, suppress=True, threshold=100000, linewidth=1000)
        # print(j_auto[:16, 32:])
        # # print(j_fd[:16, 32:])
        # print(j_man[:16, 32:])
        # # print((j_auto-j_fd)[:16, 32:])
        # # print((j_auto-j_man)[:16, 32:])
        # print(np.abs(j_auto-j_man).max())
        # print(np.abs(j_auto-j_fd).max())
        assert np.allclose(j_auto, j_man)

        sol = solver.solve(maxiters=10)

        split_sols = mesh.split_quantities(sol)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(8*3, 6))
        # plt_objs = mesh.plot(
        #     [v[:, None] for v in exact_vel_vals.T]+[exact_pres_vals])
        # plt_objs = mesh.plot(split_sols, axs=axs)
        exact_sols = [v[:, None] for v in exact_vel_vals.T]+[exact_pres_vals]
        # plt_objs = mesh.plot(
        #     [v-u for v, u in zip(exact_sols, split_sols)])
        # for ax, obj in zip(axs, plt_objs):
        #     plt.colorbar(obj, ax=ax)
        #     plt.show()

        # check value used to enforce unique pressure is found correctly
        assert np.allclose(
            split_sols[-1][pres_idx], pres_val)

        for exact_v, v in zip(exact_vel_vals.T, split_sols[:-1]):
            assert np.allclose(exact_v, v[:, 0])
            print(np.abs(exact_pres_vals-split_sols[-1]).max())
            assert np.allclose(exact_pres_vals, split_sols[-1], atol=7e-8)

    def test_stokes_solver_mms(self):
        s0, depth, L, alpha = 2, .1, 1, 1e-1
        mesh_transforms = get_vertical_2d_mesh_transforms_from_string(
            [-L, L], f"{s0}-{alpha}*x**2-{depth}", f"{s0}-{alpha}*x**2")
        test_cases = [
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], False],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["N", "D"], False],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "D"], True],
            [[0, 1], [4], ["(1-x)**2"], "x**2", ["D", "N"], True],
            [[0, 1, 0, 1], [20, 20],
             ["-cos(pi*x)*sin(pi*y)", "sin(pi*x)*cos(pi*y)"], "x**3*y**3",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [6, 7],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], False],
            [[0, 1, 0, 1], [4, 4], #[12, 12],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], True],
            [[0, 1, 0, 1], [8, 8],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "D"], True, mesh_transforms],
            [[0, 1, 0, 1], [8, 8],
             ["16*x**2*(1-x)**2*y**2", "20*x*(1-x)*y*(1-y)"], "x**1*y**2",
             ["D", "D", "D", "N"], True, mesh_transforms]
        ]
        for test_case in test_cases:
            self._check_stokes_solver_mms(*test_case)

    def _check_shallow_ice_solver_mms(
            self, domain_bounds, orders, depth_string, bed_string, beta_string,
            bndry_types, A, rho, n, g, transient):
        nphys_vars = len(orders)
        depth_fun, bed_fun, beta_fun, forc_fun, flux_funs = (
            setup_shallow_ice_manufactured_solution(
                depth_string, bed_string, beta_string, A, rho, n,
                g, nphys_vars, transient))

        depth_fun = Function(depth_fun)
        bed_fun = Function(bed_fun)
        beta_fun = Function(beta_fun)
        forc_fun = Function(forc_fun)
        flux_funs = Function(flux_funs)

        bndry_conds = _get_boundary_funs(
            nphys_vars, bndry_types, depth_fun, flux_funs)
        mesh = CartesianProductCollocationMesh(
            domain_bounds, orders)

        solver = SteadyStatePDE(ShallowIce(
            mesh, bndry_conds, bed_fun, beta_fun, forc_fun, A, rho, n, g))

        exact_sol = depth_fun(mesh.mesh_pts)
        print(np.abs(solver.residual._raw_residual(exact_sol[:, 0])[0]).max())
        print(np.abs(solver.residual._raw_residual(exact_sol[:, 0])))

        def fun(s):
            return solver.residual._raw_residual(torch.as_tensor(s)).numpy()
        j_fd = approx_jacobian(fun, exact_sol[:, 0].numpy())
        # j_man = solver.residual._raw_residual(torch.as_tensor(exact_sol[:, 0]))[1].numpy()
        j_auto = torch.autograd.functional.jacobian(
            lambda s: solver.residual._raw_residual(s),
            exact_sol[:, 0].clone().requires_grad_(True), strict=True).numpy()
        j_man = solver.residual._raw_jacobian(exact_sol[:, 0].clone()).numpy()
        np.set_printoptions(precision=2, suppress=True, threshold=100000, linewidth=1000)
        print(j_fd)
        print(j_auto)
        print(j_man)
        assert np.allclose(j_auto, j_man)
        
        assert np.allclose(
            solver.residual._raw_residual(exact_sol[:, 0])[0], 0, atol=2e-8)
        assert np.allclose(
            solver.residual._residual(exact_sol[:, 0])[0], 0, atol=2e-8)



    def test_shallow_ice_solver_mms(self):
        s0, depth, alpha = 2, .1, 1e-1
        test_cases = [
            # [[-1, 1], [4], "1", f"{s0}-{alpha}*x**2-{depth}", "1",
            #  ["D", "D"], 1, 1, 1, 1, False]
            [[-1, 1, -1, 1], [4, 4], "1", f"{s0}-{alpha}*x**2-{depth}-(1+y)", "1",
             ["D", "D", "D", "D"], 1, 1, 1, 1, False]
        ]
        for test_case in test_cases:
            self._check_shallow_ice_solver_mms(*test_case)





if __name__ == "__main__":
    manual_pde_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestManualPDE)
    unittest.TextTestRunner(verbosity=2).run(manual_pde_test_suite)
