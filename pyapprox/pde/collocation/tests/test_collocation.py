import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution
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
            basis:  OrthogonalCoordinateCollocationBasis,
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
        return self._bkd.sum(
            self._mesh_bndry.normals(pts)*self._flux_fun(pts), axis=1
        )

    def _manufactured_callable(self, pts):
        return (
            self._alpha*self._sol_fun(pts)[:, 0]
            + self._beta*self._robin_normal_flux(pts)
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
            sol = ScalarSolutionFromCallable(basis, lambda x: sol_fun(x)[:, 0])
            bndry_funs.append(
                DirichletBoundaryFromFunction(mesh_bndry, sol)
            )
        return bndry_funs

    def _setup_robin_boundary_conditions(
            self,
            basis: OrthogonalCoordinateCollocationBasis,
            sol_fun: callable,
            flux_fun: callable,
    ):
        # use the same alpha, beta for every test
        alpha, beta = 2.0, 3.0
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

    def _check_steady_state_advection_diffusion_reaction(
            self, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, basis,
            nl_diff_funs=[None, None]):
        bkd = self.get_backend()
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], False,
                nl_diff_funs[0], bkd=bkd
            )
        )
        # TODO make function take in callable. Then call set_values
        # in steadystatepde on mesh_pts and similarly for transient PDE
        exact_sol = ScalarSolutionFromCallable(
            basis, lambda x: sol_fun(x)[:, 0]
        )

        # test plot runs
        ax = plt.figure().gca()
        exact_sol.plot(ax, 51)

        diffusion = ImutableScalarFunctionFromCallable(
            basis, lambda x: diff_fun(x)[:, 0]
        )
        forcing = ImutableScalarFunctionFromCallable(
            basis, lambda x: forc_fun(x)[:, 0]
        )

        # hack
        react_fun = lambda x: bkd.ones(x.T.shape)
        reaction = ImutableScalarFunctionFromCallable(
            basis, lambda x: react_fun(x)[:, 0]
        )
        physics = LinearReactionDiffusionEquation(forcing, diffusion, reaction)
        np.set_printoptions(linewidth=1000)
        # physics = LinearDiffusionEquation(forcing, diffusion)
        residual = physics.residual(exact_sol)
        print(residual.get_values()[0, 0])
        assert bkd.allclose(
            residual.get_values()[0, 0], bkd.zeros(exact_sol.nmesh_pts(),)
        )

        # boundaries = self._setup_dirichlet_boundary_conditions(
        #     basis, sol_fun
        # )
        # boundaries = self._setup_robin_boundary_conditions(
        #     basis, sol_fun, flux_funs
        # )
        boundaries = self._setup_periodic_boundary_conditions(
            basis
        )
        physics.set_boundaries(boundaries)
        solver = SteadyStatePDE(physics, NewtonSolver(verbosity=2, maxiters=1))
        init_sol = ScalarSolutionFromCallable(
            basis, lambda x: bkd.ones(x.shape[1],)
        )
        sol = solver.solve(init_sol)
        # print(sol.get_values()[0, 0])
        # print(exact_sol.get_values()[0, 0])

        # consider renaming get_values with get_matrix_values and
        # using get_values for returning values of size consistent with number
        # of rows and columns, for example scalarfunction just returns a
        # single 1d vector
        assert bkd.allclose(
            sol.get_values(), exact_sol.get_values()
        )

    def test_advection_diffusion_reaction(self):
        bkd = self.get_backend()
        test_cases = [
            # ["-(x-1)*x/2", "4", ["0"], [None, None], ["D", "D"],
            #  self._setup_cheby_basis_1d([5], [0, 1])
            #  ],
            ["sin(x)", "1", ["0"],
             [lambda sol: 1*sol, lambda sol: bkd.ones(sol.shape[0])], ["D", "D"],
             self._setup_cheby_basis_1d([20], [0, 2*np.pi])
             ],
            # ["x**2*y**2", "2", ["0", "0"], [None, None], ["D", "D", "D", "D"],
            #  self._setup_rect_cheby_basis_2d([4, 4], [0, 1, 0, 1])
            #  ],
            # ["x**2*y**2*z**2", "2", ["0", "0", "0"], [None, None],
            #  ["D", "D", "D", "D", "D", "D"],
            #  self._setup_cube_cheby_basis_3d([3, 3, 3], [0, 1, 0, 1, 0, 1])
            #  ],
        ]
        for test_case in test_cases:
            self._check_steady_state_advection_diffusion_reaction(*test_case)


class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCollocation(TestCollocation, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
