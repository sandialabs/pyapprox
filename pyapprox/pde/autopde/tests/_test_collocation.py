import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution
)
from pyapprox.pde.autopde._collocationbasis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
    ChebyshevCollocationBasis3D,
    nabla,
    LinearDiffusionEquation,
    LinearReactionDiffusionEquation,
    Function,
    ImutableScalarFunctionFromCallable,
    DirichletBoundaryFromFunction,
    RobinBoundary,
    RobinBoundaryFromFunction,
    PeriodicBoundary,
    ScalarSolutionFromCallable,
    OrthogonalCoordinateCollocationBasis,
    OrthogonalCoordinateMeshBoundary,
)
from pyapprox.pde.autopde._mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
    ScaleAndTranslationTransform3D,
)
from pyapprox.pde.autopde._mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
    ChebyshevCollocationMesh3D,
    OrthogonalCoordinateMesh,
)
from pyapprox.pde.autopde._solvers import SteadyStatePDE, NewtonSolver


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

    def test_scalar_differential_operators_1d(self):
        bkd = self.get_backend()
        bounds = [0, 1]
        nterms = [5]
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)

        def test_fun(xx):
            return (xx.T)**3

        def test_grad(xx):
            return 3 * (xx.T)**2

        fun_values = test_fun(basis.mesh.mesh_pts())
        jac = "identity"
        fun = ScalarFunction(
            basis, fun_values[:, 0], jac=jac
        )

        plot_samples = bkd.linspace(*bounds, 101)[None, :]
        assert bkd.allclose(fun(plot_samples), test_fun(plot_samples))

        # check plots run without calling plt.show
        ax = plt.figure().gca()
        ax.plot(plot_samples[0], test_fun(plot_samples), '-k')
        fun.plot(ax, ls='--', color='r')

        gradfun = nabla(fun)
        assert np.allclose(
            gradfun(plot_samples)[:, 0, :].T, test_grad(plot_samples)
        )

        ax = plt.figure().gca()
        ax.plot(plot_samples[0], test_grad(plot_samples), '-k')
        ax.plot(
            plot_samples[0], gradfun(plot_samples)[0, 0, :], ls='--', color='r'
        )

    def test_scalar_differential_operators_2d(self):
        bkd = self.get_backend()
        bounds = [0, 1, 0, 1]
        nterms = [4, 4]
        transform = ScaleAndTranslationTransform2D([-1, 1, -1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh2D(nterms, transform)
        basis = ChebyshevCollocationBasis2D(mesh)

        def test_fun(xx):
            return bkd.sum(xx**3, axis=0)[:, None]

        def test_grad(xx):
            return 3 * xx ** 2

        fun_values = test_fun(basis.mesh.mesh_pts())
        # fun is independent of the solution
        fun = ScalarFunction(basis, fun_values[:, 0], jac="zero")

        X, Y, plot_samples = get_meshgrid_samples([0, 1, 0, 1], 11, bkd=bkd)
        assert bkd.allclose(
           fun(plot_samples)[0, 0, :, None], test_fun(plot_samples)
        )

        # check plots run without calling plt.show
        ax = plt.figure().gca()
        fun.plot(ax)
        # plt.show()

        gradfun = nabla(fun)
        assert np.allclose(
            gradfun(plot_samples)[:, 0, :], test_grad(plot_samples)
        )
        assert bkd.allclose(
            gradfun.get_jacobian(), bkd.zeros(gradfun.jacobian_shape())
        )

        # fun is the solution
        fun = ScalarFunction(
            basis, fun_values[:, 0], jac="identity"
        )
        gradfun = nabla(fun)
        for ii in range(fun.nphys_vars()):
            assert bkd.allclose(
                gradfun.get_jacobian()[ii, 0], basis._deriv_mats[ii]
            )

        # fun is a function of the solution
        jac = bkd.diag(4*fun_values[:, 0]**3)
        fun = ScalarFunction(
            basis, fun_values[:, 0]**4, jac=jac
        )
        gradfun = nabla(fun)
        for ii in range(fun.nphys_vars()):
            assert bkd.allclose(
                gradfun.get_jacobian()[ii, 0],
                basis._deriv_mats[ii] @ bkd.diag(4*fun_values[:, 0]**3)
            )

        if not bkd.jacobian_implemented():
            return

        def jacfun(fun_values):
            fun = ScalarFunction(
                basis, fun_values**4, jac=jac
            )
            gradfun = nabla(fun)
            return gradfun.get_values()

        assert bkd.allclose(
            gradfun.get_jacobian(),
            bkd.jacobian(jacfun, fun_values[:, 0])
        )

        def test_gfun(xx):
            return bkd.sum(xx**2, axis=0)[:, None]

        # TODO once fix failing test with jac='zero' for gfun
        # add test with jac being nonzero for gfun
        gfun = ScalarFunction(
            basis, test_gfun(basis.mesh.mesh_pts())[:, 0], jac="zero"
        )
        prodfun = gfun*nabla(fun)
        for ii in range(prodfun.nphys_vars()):
            assert bkd.allclose(
                test_gfun(basis.mesh.mesh_pts())[:, 0]*nabla(fun).get_values()[ii],
                prodfun.get_values()[ii]
            )

        def jacprodfun(fun_values):
            fun = ScalarFunction(
                basis, fun_values**4, jac=jac
            )
            gfun = ScalarFunction(
                basis, test_gfun(basis.mesh.mesh_pts())[:, 0], jac="zero"
            )
            prodfun = gfun*nabla(fun)
            return prodfun.get_values()

        assert bkd.allclose(
             prodfun.get_jacobian(),
             bkd.jacobian(jacprodfun, fun_values[:, 0])
        )

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
            sol.get_values()[0, 0], exact_sol.get_values()[0, 0]
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
