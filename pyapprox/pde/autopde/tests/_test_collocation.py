import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.autopde.manufactured_solutions import (
    setup_advection_diffusion_reaction_manufactured_solution
)
from pyapprox.pde.autopde._collocationbasis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
)
from pyapprox.pde.autopde._mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
)
from pyapprox.pde.autopde._mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
)
from pyapprox.pde.autopde._collocationbasis import (
    ScalarCollocationFunction, nabla, div, laplace, LinearDiffusionEquation
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
        fun = ScalarCollocationFunction(
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
        fun = ScalarCollocationFunction(basis, fun_values[:, 0], jac="zero")

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
        fun = ScalarCollocationFunction(
            basis, fun_values[:, 0], jac="identity"
        )
        gradfun = nabla(fun)
        for ii in range(fun.nphys_vars()):
            assert bkd.allclose(
                gradfun.get_jacobian()[ii, 0], basis._deriv_mats[ii]
            )

        # fun is a function of the solution
        jac = bkd.diag(4*fun_values[:, 0]**3)
        fun = ScalarCollocationFunction(
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
            fun = ScalarCollocationFunction(
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

        #TODO once fix failing test with jac='zero' for gfun
        # add test with jac being nonzero for gfun
        gfun = ScalarCollocationFunction(
            basis, test_gfun(basis.mesh.mesh_pts())[:, 0], jac="zero"
        )
        prodfun = gfun*nabla(fun)
        for ii in range(prodfun.nphys_vars()):
            assert bkd.allclose(
                test_gfun(basis.mesh.mesh_pts())[:, 0]*nabla(fun).get_values()[ii],
                prodfun.get_values()[ii]
            )

        def jacprodfun(fun_values):
            fun = ScalarCollocationFunction(
                basis, fun_values**4, jac=jac
            )
            gfun = ScalarCollocationFunction(
                basis, test_gfun(basis.mesh.mesh_pts())[:, 0], jac="zero"
            )
            prodfun = gfun*nabla(fun)
            return prodfun.get_values()

        assert bkd.allclose(
             prodfun.get_jacobian(),
             bkd.jacobian(jacprodfun, fun_values[:, 0])
        )

    def _setup_cheby_basis_1d(self, nterms, bounds):
        bkd = self.get_backend()
        transform = ScaleAndTranslationTransform1D([-1, 1], bounds, bkd)
        mesh = ChebyshevCollocationMesh1D(nterms, transform)
        basis = ChebyshevCollocationBasis1D(mesh)
        return basis

    def _check_steady_state_advection_diffusion_reaction(
            self, sol_string, diff_string, vel_strings,
            react_funs, bndry_types, basis,
            nl_diff_funs=[None, None]):
        bkd = self.get_backend()
        sol_fun, diff_fun, vel_fun, forc_fun, flux_funs = (
            setup_advection_diffusion_reaction_manufactured_solution(
                sol_string, diff_string, vel_strings, react_funs[0], False,
                nl_diff_funs[0]))

        exact_sol = ScalarCollocationFunction(
            basis, sol_fun(basis.mesh.mesh_pts())[:, 0], "identity"
        )
        diffusion = ScalarCollocationFunction(
            basis, diff_fun(basis.mesh.mesh_pts())[:, 0], "zero"
        )
        forcing = ScalarCollocationFunction(
            basis, forc_fun(basis.mesh.mesh_pts())[:, 0], "zero"
        )
        physics = LinearDiffusionEquation(diffusion, forcing)
        residual, jac = physics.residual(exact_sol)
        assert bkd.allclose(residual, bkd.zeros(exact_sol.nmesh_pts(),))

    def test_advection_diffusion_reaction(self):
        test_cases = [
            ["-(x-1)*x/2", "4", ["0"], [None, None], ["D", "D"],
             self._setup_cheby_basis_1d([5], [0, 1])
             ],
            # todo add 2d test
            # [[0, 1, 0, 1], [4, 4], "y**2*x**2", "1", ["0", "0"],
            #  [lambda sol: 0*sol,
            #   lambda sol: torch.zeros((sol.shape[0],))],
        ]
        for test_case in test_cases:
            self._check_steady_state_advection_diffusion_reaction(*test_case)


class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
