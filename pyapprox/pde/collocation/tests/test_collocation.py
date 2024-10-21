import unittest
import itertools
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution,
)
from pyapprox.pde.collocation.manufactured_solutions import (
    Diffusion, ReactionDiffusion,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
    ChebyshevCollocationBasis3D,
    OrthogonalCoordinateCollocationBasis,
)
from pyapprox.pde.collocation.physics import (
    LinearDiffusionEquation,
    LinearReactionDiffusionEquation,
)
from pyapprox.pde.collocation.functions import (
    ImutableScalarFunctionFromCallable,
    ScalarSolutionFromCallable,
)

from pyapprox.pde.collocation.boundaryconditions import (
    DirichletBoundaryFromFunction,
    RobinBoundary,
    RobinBoundaryFromFunction,
    PeriodicBoundary,
    OrthogonalCoordinateMeshBoundary,
)
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
    ScaleAndTranslationTransform3D,
)
from pyapprox.pde.collocation.mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
    ChebyshevCollocationMesh3D,
)
from pyapprox.pde.collocation.solvers import SteadyStatePDE, NewtonSolver


class RobinBoundaryFromManufacturedSolution(RobinBoundaryFromFunction):
    def __init__(
        self,
        mesh_bndry: OrthogonalCoordinateMeshBoundary,
        alpha: float,
        beta: float,
        basis: OrthogonalCoordinateCollocationBasis,
        sol_fun: callable,
        flux_fun: callable,
    ):
        self._basis = basis
        self._sol_fun = sol_fun
        self._flux_fun = flux_fun
        self._bkd = self._basis._bkd
        # must set alpha before calling set function
        # because manufactured_callable with be called and needs alpha
        RobinBoundary.__init__(self, mesh_bndry, alpha, beta)
        self._set_function(
            ImutableScalarFunctionFromCallable(
                basis, self._manufactured_callable
            )
        )

    def _robin_normal_flux(self, pts):
        # normal_fun =  partial(transform.normal, ii)
        # normal_vals = torch.as_tensor(normal_fun(xx), dtype=torch.double)
        # flux_vals = torch.as_tensor(flux_funs(xx), dtype=torch.double)
        flux_vals = self._flux_fun(pts)
        if flux_vals.ndim != 2:
            raise ValueError("flux_fun must return 2d array")
        return self._bkd.sum(
            self._mesh_bndry.normals(pts) * self._flux_fun(pts), axis=1
        )

    def _manufactured_callable(self, pts):
        return (
            self._alpha * self._sol_fun(pts)
            + self._beta * self._robin_normal_flux(pts)
        )


class TestCollocation:
    def setUp(self):
        np.random.seed(1)

    def _setup_cheby_basis_1d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)
        return basis

    def _setup_rect_cheby_basis_2d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        transform = ScaleAndTranslationTransform2D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_cube_cheby_basis_3d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        transform = ScaleAndTranslationTransform3D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh3D(nterms, transform)
        basis = ChebyshevCollocationBasis3D(mesh)
        return basis

    def _setup_dirichlet_boundary_conditions(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        sol_fun: callable,
    ):
        bndry_funs = []
        for bndry_name, mesh_bndry in basis.mesh.get_boundaries().items():
            sol = ScalarSolutionFromCallable(basis, sol_fun)
            bndry_funs.append(DirichletBoundaryFromFunction(mesh_bndry, sol))
        return bndry_funs

    def _setup_robin_boundary_conditions(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        sol_fun: callable,
        flux_fun: callable,
    ):
        # use the same alpha, beta for every test
        alpha, beta = 2.0, 3.0
        # alpha, beta = 1.0, 1.0
        bndry_funs = []
        for bndry_name, mesh_bndry in basis.mesh.get_boundaries().items():
            bndry_funs.append(
                RobinBoundaryFromManufacturedSolution(
                    mesh_bndry, alpha, beta, basis, sol_fun, flux_fun
                )
            )
        return bndry_funs

    def _setup_periodic_boundary_conditions(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
    ):
        bndry_funs = []
        boundaries = basis.mesh.get_boundaries()
        mesh_bndry = boundaries[list(boundaries.keys())[0]]
        bndry_pair_names_dict = mesh_bndry.names_of_boundary_pairs()
        for bndry_name, mesh_bndry in boundaries.items():
            if bndry_name not in bndry_pair_names_dict:
                continue
            partner_mesh_bndry = boundaries[bndry_pair_names_dict[bndry_name]]
            bndry_funs.append(
                PeriodicBoundary(mesh_bndry, partner_mesh_bndry, basis)
            )
        return bndry_funs

    def _setup_boundary_conditions(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            sol_fun: callable,
            flux_fun: callable,
            bndry_types: str,
    ):
        if bndry_types == "D":
            return self._setup_dirichlet_boundary_conditions(basis, sol_fun)
        if bndry_types == "R":
            return self._setup_robin_boundary_conditions(
                basis, sol_fun, flux_fun
            )
        if bndry_types == "P":
            return self._setup_periodic_boundary_conditions(basis)
        if bndry_types != "M":
            # M = Mixed:
            raise ValueError("incorrect bndry_type specified")
        # mix robin dirichlet boundaries
        boundaries = basis.mesh.get_boundaries()
        sol = ScalarSolutionFromCallable(basis, sol_fun)
        mesh_bndry = boundaries["left"]
        alpha, beta = 2., 3.
        bndry_funs = [
            RobinBoundaryFromManufacturedSolution(
                mesh_bndry, alpha, beta, basis, sol_fun, flux_fun
            )
        ]
        for bndry_name, mesh_bndry in boundaries.items():
            if bndry_name == "left":
                continue
            bndry_funs.append(DirichletBoundaryFromFunction(mesh_bndry, sol))
        print(bndry_funs)
        return bndry_funs

    def _check_steady_state_advection_diffusion_reaction(
        self,
        sol_string : str,
        diff_string: str,
        vel_strings: str,
        react_tup: Tuple[str, callable],
        bndry_types: str,
        basis : OrthogonalCoordinateCollocationBasis,
    ):
        bkd = self.get_backend()
        react_str, react_fun = react_tup
        man_sol = ReactionDiffusion(
            len(vel_strings), sol_string, diff_string, react_str, bkd=bkd,
            oned=True
        )
        print(man_sol)

        # TODO make function take in callable. Then call set_values
        # in steadystatepde on mesh_pts and similarly for transient PDE
        exact_sol = ScalarSolutionFromCallable(
            basis, man_sol.functions["solution"]
        )

        # test plot runs
        ax = exact_sol.get_plot_axis()
        exact_sol.plot(ax, 51)

        diffusion = ImutableScalarFunctionFromCallable(
            basis, man_sol.functions["diffusion"]
        )
        forcing = ImutableScalarFunctionFromCallable(
            basis, man_sol.functions["forcing"]
        )
        # todo man_sol.functions["reaction"] contains the values u
        # but linearreactiondiffusionequations needs 1 from (1*u)
        reaction = ImutableScalarFunctionFromCallable(
            basis, react_fun,
        )
        physics = LinearReactionDiffusionEquation(forcing, diffusion, reaction)
        # physics = LinearDiffusionEquation(forcing, diffusion)
        residual = physics.residual(exact_sol)
        # np.set_printoptions(linewidth=1000)
        # print(residual.get_values()[0, 0])
        assert bkd.allclose(
            residual.get_values()[0, 0],
            bkd.zeros(
                exact_sol.nmesh_pts(),
            ),
        )

        boundaries = self._setup_boundary_conditions(
            basis,
            man_sol.functions["solution"],
            man_sol.functions["flux"],
            bndry_types,
        )
        physics.set_boundaries(boundaries)
        solver = SteadyStatePDE(physics, NewtonSolver(verbosity=2, maxiters=1))
        init_sol = ScalarSolutionFromCallable(
            basis,
            lambda x: bkd.ones(
                x.shape[1],
            ),
        )
        sol = solver.solve(init_sol)
        # print(sol.get_values())
        # print(exact_sol.get_values())
        assert bkd.allclose(sol.get_values(), exact_sol.get_values())

    def test_steady_advection_diffusion_reaction_1D(self):
        bkd = self.get_backend()
        test_case_args = [
            ["-(x-1)*x/2"],   # sol_string
            ["4", "(x+1)"],   # diff_string
            [["0"], ["1"]],   # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1], ), 2.)]
            ],     # react_str
            ["D", "R", "M"],  # bndry_types
            [
                self._setup_cheby_basis_1d([5], [0, 1]),
                self._setup_cheby_basis_1d([5], [0, 2])
            ],  # basis
        ]
        for test_case in itertools.product(*test_case_args):
            print(test_case)
            self._check_steady_state_advection_diffusion_reaction(*test_case)

        # test periodic BCs which requires a periodic sol_str and reaction term
        # to make solution unique
        test_case = [
            "sin(x)",
            "1",
            ["0"],
            # warning 1*u will have zero as a trivial solution
            ["2*u", lambda x: bkd.full((x.shape[1], ), 2.)],
            "P",
            self._setup_cheby_basis_1d([20], [0, 2 * np.pi]),
            ]
        self._check_steady_state_advection_diffusion_reaction(*test_case)

    def test_steady_advection_diffusion_reaction_2D(self):
        bkd = self.get_backend()
        test_case_args = [
            ["x**2*y**2"],   # sol_string
            ["4", "(x+1)"],   # diff_string
            [["0", "0"], ["1", "2"]],   # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1], ), 2.)]
            ],     # react_str
            ["D", "R", "M"],  # bndry_types
            [self._setup_rect_cheby_basis_2d([5, 5], [0, 1, 0, 2]),],  # basis
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_steady_state_advection_diffusion_reaction(*test_case)

        # test periodic BCs which requires a periodic sol_str and reaction term
        # to make solution unique
        test_case = [
            "sin(x)*sin(y)",
            "1",
            ["0", "0"],
            # warning 1*u will have zero as a trivial solution
            ["2*u", lambda x: bkd.full((x.shape[1], ), 2.)],
            "P",
            self._setup_rect_cheby_basis_2d(
                [20, 20], [0, 2 * np.pi, 0, 2 * np.pi]
            ),
            ]

    def test_steady_advection_diffusion_reaction_3D(self):
        bkd = self.get_backend()
        test_case_args = [
            ["x**2*y**2*z**2"],   # sol_string
            ["4", "(x+1)"],   # diff_string
            [["0", "0", "0"], ["1", "2", "3"]],   # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1], ), 2.)]
            ],     # react_str
            ["D", "R", "M"],  # bndry_types
            # basis
            [self._setup_cube_cheby_basis_3d([5, 5, 5], [0, 1, 0, 2, -1, 1]),],
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_steady_state_advection_diffusion_reaction(*test_case)


class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCollocation(TestCollocation, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
