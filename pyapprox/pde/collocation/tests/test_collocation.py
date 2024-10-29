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
    ManufacturedSolution,
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.basis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
    ChebyshevCollocationBasis3D,
    OrthogonalCoordinateCollocationBasis,
)
from pyapprox.pde.collocation.physics import (
    # LinearDiffusionEquation,
    LinearReactionDiffusionEquation,
)
from pyapprox.pde.collocation.functions import (
    ImutableScalarFunctionFromCallable,
    ScalarSolutionFromCallable,
    ImutableScalarTransientFunctionFromCallable,
    ScalarTransientSolutionFromCallable,
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
    FixedScaleAndTranslationTransform3D,
    PolarTransform,
    SphericalTransform,
    CompositionTransform,
)
from pyapprox.pde.collocation.mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
    ChebyshevCollocationMesh3D,
)
from pyapprox.pde.collocation.solvers import SteadyPDE, NewtonSolver


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
        return self._alpha * self._sol_fun(
            pts
        ) + self._beta * self._robin_normal_flux(pts)


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

    def _setup_disc_cheby_basis_2d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        scale_transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, bkd
        )
        polar_transform = PolarTransform(bkd)
        transform = CompositionTransform([scale_transform, polar_transform])
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_cube_cheby_basis_3d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        transform = ScaleAndTranslationTransform3D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh3D(nterms, transform)
        basis = ChebyshevCollocationBasis3D(mesh)
        return basis

    def _setup_sphere_cheby_basis_3d(self, nterms: list, bounds: list):
        bkd = self.get_backend()
        scale_trans = FixedScaleAndTranslationTransform3D(
            [-1, 1, -1, 1, -1, 1], bounds, bkd
        )
        sphere_trans = SphericalTransform(bkd)
        transform = CompositionTransform([scale_trans, sphere_trans])
        mesh = ChebyshevCollocationMesh3D(nterms, transform)
        basis = ChebyshevCollocationBasis3D(mesh)
        return basis

    def _setup_dirichlet_boundary_conditions(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        man_sol: ManufacturedSolution,
    ):
        bndry_funs = []
        for bndry_name, mesh_bndry in basis.mesh.get_boundaries().items():
            sol = self._setup_scalar_solution_from_manufactured_solution(
                basis, man_sol
            )
            bndry_funs.append(DirichletBoundaryFromFunction(mesh_bndry, sol))
        return bndry_funs

    def _setup_robin_boundary_conditions(
        self,
        basis: OrthogonalCoordinateCollocationBasis,
        man_sol: ManufacturedSolution,
    ):
        # use the same alpha, beta for every test
        alpha, beta = 2.0, 3.0
        # alpha, beta = 1.0, 1.0
        bndry_funs = []
        for bndry_name, mesh_bndry in basis.mesh.get_boundaries().items():
            bndry_funs.append(
                RobinBoundaryFromManufacturedSolution(
                    mesh_bndry,
                    alpha,
                    beta,
                    basis,
                    man_sol.functions["solution"],
                    man_sol.functions["flux"],
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
        bndry_types: str,
        man_sol: ManufacturedSolution,
    ):
        if bndry_types == "D":
            return self._setup_dirichlet_boundary_conditions(basis, man_sol)
        if bndry_types == "R":
            return self._setup_robin_boundary_conditions(
                basis, man_sol
            )
        if bndry_types == "P":
            return self._setup_periodic_boundary_conditions(basis)
        if bndry_types != "M":
            # M = Mixed:
            raise ValueError("incorrect bndry_type specified")
        # mix robin dirichlet boundaries
        boundaries = basis.mesh.get_boundaries()
        mesh_bndry = boundaries["left"]
        alpha, beta = 2.0, 3.0
        bndry_funs = [
            RobinBoundaryFromManufacturedSolution(
                mesh_bndry,
                alpha,
                beta,
                basis,
                man_sol.functions["solution"],
                man_sol.functions["flux"],
            )
        ]
        for bndry_name, mesh_bndry in boundaries.items():
            if bndry_name == "left":
                continue
            sol = self._setup_scalar_solution_from_manufactured_solution(
                basis, man_sol
            )
            bndry_funs.append(DirichletBoundaryFromFunction(mesh_bndry, sol))
        return bndry_funs

    def _setup_scalar_solution_from_manufactured_solution(
            self, basis, man_sol
    ):
        name = "solution"
        if not man_sol.transient[name]:
            return ScalarSolutionFromCallable(
                basis, man_sol.functions[name]
            )
        return ScalarTransientSolutionFromCallable(
            basis, man_sol.functions[name]
        )

    def _setup_immutable_function_from_manufactured_solution(
            self, name, basis, man_sol
    ):
        if not man_sol.transient[name]:
            return ImutableScalarFunctionFromCallable(
                basis, man_sol.functions[name]
            )
        return ImutableScalarTransientFunctionFromCallable(
            basis, man_sol.functions[name]
        )

    def _check_steady_state_advection_diffusion_reaction(
        self,
        sol_string: str,
        diff_string: str,
        vel_strings: str,
        react_tup: Tuple[str, callable],
        bndry_types: str,
        basis: OrthogonalCoordinateCollocationBasis,
    ):
        bkd = self.get_backend()
        react_str, react_fun = react_tup
        man_sol = AdvectionDiffusionReaction(
            len(vel_strings),
            sol_string,
            diff_string,
            react_str,
            vel_strings,
            bkd=bkd,
            oned=True,
        )
        print(man_sol)
        print(man_sol.transient, sol_string)

        # TODO make function take in callable. Then call set_values
        # in steadystatepde on mesh_pts and similarly for transient PDE
        exact_sol = self._setup_scalar_solution_from_manufactured_solution(
            basis, man_sol
        )
        print(exact_sol)

        # test plot runs
        fig, ax = exact_sol.get_plot_axis()
        basis.mesh.plot(ax)
        exact_sol.plot(ax, 101, fig=fig)
        ax.set_aspect("equal")
        # plt.show()

        # diffusion = ImutableScalarFunctionFromCallable(
        #     basis, man_sol.functions["diffusion"]
        # )
        diffusion = self._setup_immutable_function_from_manufactured_solution(
            "diffusion", basis, man_sol
        )
        forcing = self._setup_immutable_function_from_manufactured_solution(
            "forcing", basis, man_sol
        )
        # todo man_sol.functions["reaction"] contains the values u
        # but linearreactiondiffusionequations needs 1 from (1*u)
        reaction = ImutableScalarFunctionFromCallable(
            basis,
            react_fun,
        )
        physics = LinearReactionDiffusionEquation(forcing, diffusion, reaction)
        # physics = LinearDiffusionEquation(forcing, diffusion)
        residual = physics.residual(exact_sol)
        # np.set_printoptions(linewidth=1000)
        print(residual.get_values()[0, 0])
        print(bkd.abs(residual.get_values()[0, 0]).max())
        assert bkd.allclose(
            residual.get_values()[0, 0],
            bkd.zeros(
                exact_sol.nmesh_pts(),
            ),
        )

        boundaries = self._setup_boundary_conditions(
            basis,
            bndry_types,
            man_sol,
        )
        physics.set_boundaries(boundaries)
        solver = SteadyPDE(physics, NewtonSolver(verbosity=2, maxiters=1))
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
            ["-(x-1)*x/2"],  # sol_string
            ["4", "(x+1)"],  # diff_string
            [["0"], ["1"]],  # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1],), 2.0)],
            ],  # react_str
            ["D", "R", "M"],  # bndry_types
            [
                self._setup_cheby_basis_1d([5], [0, 1]),
                self._setup_cheby_basis_1d([5], [0, 2]),
            ],  # basis
        ]

        # for test_case in itertools.product(*test_case_args):
        #     print(test_case)
        #     self._check_steady_state_advection_diffusion_reaction(*test_case)

        # test periodic BCs which requires a periodic sol_str and reaction term
        # to make solution unique
        test_case = [
            "sin(x)",
            "1",
            ["0"],
            # warning 1*u will have zero as a trivial solution
            ["2*u", lambda x: bkd.full((x.shape[1],), 2.0)],
            "P",
            self._setup_cheby_basis_1d([20], [0, 2 * np.pi]),
        ]
        # self._check_steady_state_advection_diffusion_reaction(*test_case)

        # transient test cases
        test_case_args = [
            ["-(x-1)*x/2*(1+T)"],  # sol_string
            ["4"],  # diff_string
            [["0"]],  # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
            ],  # react_str
            ["D"],  # bndry_types
            [
                self._setup_cheby_basis_1d([5], [0, 1]),
            ],  # basis
        ]
        for test_case in itertools.product(*test_case_args):
            self._check_steady_state_advection_diffusion_reaction(*test_case)

    def test_steady_advection_diffusion_reaction_2D(self):
        bkd = self.get_backend()
        test_case_args = [
            ["x**2*y**2"],  # sol_string
            ["4", "(x+1)"],  # diff_string
            [["0", "0"], ["1", "2"]],  # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1],), 2.0)],
            ],  # react_str
            ["D", "R", "M"],  # bndry_types
            [
                self._setup_rect_cheby_basis_2d([5, 5], [0, 1, 0, 2]),
                self._setup_disc_cheby_basis_2d(
                    [30, 30], [1, 2, -np.pi / 2, np.pi / 2]
                )
            ],  # basis
        ]
        # Note for the basis: self._setup_disc_cheby_basis_2d(
        # [30, 30], [1, 2, -2 * np.pi / 4, 2 * np.pi / 4])
        # sol=(x**2+y**2) is exactly representable by a quadratic basis
        # however Del f requires computing D(cos(x)Du) which requires
        # interpolating cosine with a polynomial which requires high-degree.
        # Thus I use 30 above

        # Cannot solve PDE with r=0 because of singularity in gradient there
        # i.e. self._setup_disc_cheby_basis_2d([35, 35], [0, 1, -np.pi, np.pi])
        # need to increase 0 to small value e.g 1e-3, with dirichlet b.c.s
        # but the smaller the values the more ill conditioned the newton solve

        for test_case in itertools.product(*test_case_args):
            self._check_steady_state_advection_diffusion_reaction(*test_case)

        # test periodic BCs which requires a periodic sol_str and reaction term
        # to make solution unique
        test_case = [
            "sin(x)*sin(y)",
            "1",
            ["0", "0"],
            # warning 1*u will have zero as a trivial solution
            ["2*u", lambda x: bkd.full((x.shape[1],), 2.0)],
            "P",
            self._setup_rect_cheby_basis_2d(
                [20, 20], [0, 2 * np.pi, 0, 2 * np.pi]
            ),
        ]

    def test_steady_advection_diffusion_reaction_3D(self):
        bkd = self.get_backend()
        test_case_args = [
            ["x**2*y**2*z**2"],  # sol_string
            ["4", "(x+1)"],  # diff_string
            [["0", "0", "0"], ["1", "2", "3"]],  # vel_strings
            [
                ["0", lambda x: bkd.zeros(x.shape[1])],
                ["2*u", lambda x: bkd.full((x.shape[1],), 2.0)],
            ],  # react_str
            ["D", "R", "M"],  # bndry_types
            # basis
            [
                self._setup_cube_cheby_basis_3d(
                    [5, 5, 5], [0, 1, 0, 2, -1, 1]
                ),
            ],
        ]

        # test_case = list(itertools.product(*test_case_args))[-1]
        # test_case = [
        #     "x**2*y**2*z**2", "1", ["0", "0", "0"],
        #     ["0", lambda x: bkd.zeros(x.shape[1])], "D",
        #     self._setup_sphere_cheby_basis_3d(
        #             [1, 30, 30], [1, 1, -np.pi/2, np.pi/2, np.pi/4, np.pi/2]
        #         ),
        # ]
        # self._check_steady_state_advection_diffusion_reaction(*test_case)
        # assert False

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
