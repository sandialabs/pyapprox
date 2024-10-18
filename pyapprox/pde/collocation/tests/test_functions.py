import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.pde.collocation.basis import (
    ChebyshevCollocationBasis1D,
    ChebyshevCollocationBasis2D,
)
from pyapprox.pde.collocation.functions import (
    ScalarSolution,
    ImutableScalarFunction,
    ScalarOperatorFromCallable,
    ImutableScalarFunctionFromCallable,
)
from pyapprox.pde.collocation.physics import nabla
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform1D,
    ScaleAndTranslationTransform2D,
)
from pyapprox.pde.collocation.mesh import (
    ChebyshevCollocationMesh1D,
    ChebyshevCollocationMesh2D,
)


class TestFunctions:
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
            return (xx.T) ** 3

        def test_grad(xx):
            return 3 * (xx.T) ** 2

        fun_values = test_fun(basis.mesh.mesh_pts())
        fun = ScalarSolution(basis, fun_values[:, 0])

        # plot_samples = bkd.linspace(*bounds, 101)[None, :]
        plot_samples = bkd.linspace(*bounds, 11)[None, :]
        assert bkd.allclose(fun(plot_samples), test_fun(plot_samples)[:, 0])

        # check plots run without calling plt.show
        ax = plt.figure().gca()
        ax.plot(plot_samples[0], test_fun(plot_samples), "-k")
        fun.plot(ax, ls="--", color="r")

        gradfun = nabla(fun)
        assert np.allclose(
            gradfun(plot_samples)[:, 0, :].T, test_grad(plot_samples)
        )

        ax = plt.figure().gca()
        ax.plot(plot_samples[0], test_grad(plot_samples), "-k")
        ax.plot(
            plot_samples[0], gradfun(plot_samples)[0, 0, :], ls="--", color="r"
        )

    def _check_differential_operators_2d_with_autograd(self, basis):
        bkd = self.get_backend()

        def get_gradfun(fun_values):
            sol = ScalarSolution(basis, fun_values)

            def op_jac(vals):
                return (4 * bkd.diag(vals) ** 3)

            fun = ScalarOperatorFromCallable(
                sol,
                lambda vals: vals ** 4,
                op_jac,
            )
            gradfun = nabla(fun)
            return gradfun

        def jacfun(fun_values):
            return get_gradfun(fun_values).get_values()

        def test_fun(xx):
            return bkd.sum(xx**3, axis=0)[:, None]

        fun_values = test_fun(basis.mesh.mesh_pts())
        assert bkd.allclose(
            get_gradfun(fun_values[:, 0]).get_matrix_jacobian(),
            bkd.jacobian(jacfun, fun_values[:, 0]),
        )

        def test_gfun(xx):
            return bkd.sum(xx**2, axis=0)

        def get_prodfun(fun_values):
            gradfun = get_gradfun(fun_values)
            gfun = ImutableScalarFunctionFromCallable(basis, test_gfun)
            return gfun * gradfun

        prodfun = get_prodfun(fun_values[:, 0])
        for ii in range(prodfun.nphys_vars()):
            assert bkd.allclose(
                test_gfun(basis.mesh.mesh_pts())
                * get_gradfun(fun_values[:, 0]).get_values()[ii],
                prodfun.get_values()[ii],
            )

        def prodfun(fun_values):
            return get_prodfun(fun_values).get_values()

        assert bkd.allclose(
            get_prodfun(fun_values[:, 0]).get_matrix_jacobian(),
            bkd.jacobian(prodfun, fun_values[:, 0]),
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
            return 3 * xx**2

        fun_values = test_fun(basis.mesh.mesh_pts())
        # fun is independent of the solution
        fun = ImutableScalarFunction(basis, fun_values[:, 0])

        X, Y, plot_samples = get_meshgrid_samples([0, 1, 0, 1], 11, bkd=bkd)
        assert bkd.allclose(fun(plot_samples), test_fun(plot_samples)[:, 0])

        # check plots run without calling plt.show
        ax = plt.figure().gca()
        fun.plot(ax)
        # plt.show()

        gradfun = nabla(fun)
        assert np.allclose(
            gradfun(plot_samples)[:, 0, :], test_grad(plot_samples)
        )
        assert bkd.allclose(
            gradfun.get_matrix_jacobian(), bkd.zeros(gradfun.jacobian_shape())
        )

        # fun is the solution
        sol = ScalarSolution(basis, fun_values[:, 0])
        gradsol = nabla(sol)
        for ii in range(fun.nphys_vars()):
            assert bkd.allclose(
                gradsol.get_matrix_jacobian()[ii, 0], basis._deriv_mats[ii]
            )

        # fun is a function of the solution
        def op_jac(vals):
            return (4 * bkd.diag(vals) ** 3)

        fun = ScalarOperatorFromCallable(
            sol,
            lambda vals: vals ** 4,
            op_jac,
        )
        gradfun = nabla(fun)
        for ii in range(fun.nphys_vars()):
            assert bkd.allclose(
                gradfun.get_matrix_jacobian()[ii, 0],
                basis._deriv_mats[ii] @ bkd.diag(4 * fun_values[:, 0] ** 3),
            )

        if not bkd.jacobian_implemented():
            return

        self._check_differential_operators_2d_with_autograd(basis)


class TestNumpyFunctions(TestFunctions, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchFunctions(TestFunctions, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
