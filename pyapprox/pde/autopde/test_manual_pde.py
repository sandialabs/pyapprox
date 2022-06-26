import unittest
import torch
import numpy as np

from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
    get_vertical_2d_mesh_transforms_from_string
)
from pyapprox.pde.autopde.test_autopde import _get_boundary_funs
from pyapprox.pde.autopde.autopde import (
    CartesianProductCollocationMesh,
    TransformedCollocationMesh,
    Function, TransientFunction, SteadyStatePDE)
from pyapprox.pde.autopde.manual_pde import AdvectionDiffusionReaction


class TestAnalyticalPDE(unittest.TestCase):
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


if __name__ == "__main__":
    analytical_pde_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestAnalyticalPDE)
    unittest.TextTestRunner(verbosity=2).run(analytical_pde_test_suite)
