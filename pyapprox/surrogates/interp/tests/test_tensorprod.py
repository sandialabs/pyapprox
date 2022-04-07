import numpy as np
import unittest
from functools import partial

from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.interp.tensorprod import (
    piecewise_quadratic_interpolation,
    canonical_piecewise_quadratic_interpolation,
    get_tensor_product_piecewise_polynomial_quadrature_rule,
    tensor_product_piecewise_polynomial_interpolation
)


class TestTensorProd(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_get_tensor_product_piecewise_linear_quadrature_rule(self):
        nsamples_1d = 101

        def fun(xx):
            return np.sum(xx**2, axis=0)[:, None]

        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [-1, 1, -1, 1], 1)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 8/3.
        print(integral-true_integral)
        assert np.allclose(integral, true_integral, atol=1e-3)

        # from scipy.interpolate import griddata
        # def interp_fun(x): return griddata(xx.T, vals, x.T, method="linear")
        # from pyapprox.util.visualization import plt, get_meshgrid_function_data
        # X, Y, Z = get_meshgrid_function_data(interp_fun, [-1, 1, -1, 1], 201)
        # plt.contourf(
        #     X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 31))
        # plt.show()

        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [0, 1, 0, 2], 1)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 10./3.
        print(integral-true_integral)
        assert np.allclose(integral, true_integral, atol=1e-3)

        def fun(xx):
            return np.sum(xx**3, axis=0)[:, None]
        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [0, 1, 0, 2], 2)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 9./2.
        print(integral-true_integral)
        assert np.allclose(integral, true_integral, atol=1e-3)

    def check_piecewise_quadratic_basis(self, basis_type, levels, tol):

        samples = np.random.uniform(0, 1, (len(levels), 9))

        def fun(samples):
            # when levels is zero to test interpolation make sure
            # function is constant in that direction
            return np.sum(samples[np.array(levels) > 0]**2, axis=0)[:, None]

        interp_fun = partial(tensor_product_piecewise_polynomial_interpolation,
                             levels=levels, fun=fun, basis_type=basis_type)
        # print((fun(samples)-interp_fun(samples))/fun(samples))
        assert np.allclose(interp_fun(samples), fun(samples), rtol=tol)
        # from pyapprox import get_meshgrid_function_data, plt
        # X, Y, Z = get_meshgrid_function_data(
        #     lambda x: interp_fun(x)-fun(x), [0, 1, 0, 1], 50, qoi=0)
        # p = plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 30))
        # plt.colorbar(p)
        # print(Z.max())
        # #plt.show()

    def test_piecewise_quadratic_basis(self):
        self.check_piecewise_quadratic_basis("quadratic", [0, 1], 1e-8)
        self.check_piecewise_quadratic_basis("quadratic", [1, 1], 1e-8)
        self.check_piecewise_quadratic_basis("linear", [0, 10], 4e-4)
        self.check_piecewise_quadratic_basis("linear", [10, 10], 4e-4)

    def test_tensor_product_quadrature(self):
        num_vars = 2

        def univariate_quadrature_rule(n):
            # x, w = gauss_jacobi_pts_wts_1D(n, 0, 0)
            x, w = np.polynomial.legendre.leggauss(n)
            w *= 0.5  # not needed for gauss_jacobi_pts_wts_1D
            x = (x+1)/2.
            return x, w

        x, w = get_tensor_product_quadrature_rule(
            100, num_vars, univariate_quadrature_rule)

        def function(x): return np.sum(x**2, axis=0)
        assert np.allclose(np.dot(function(x), w), num_vars/3)

    def test_canonical_piecewise_quadratic_interpolation(self):
        num_mesh_points = 101
        mesh = np.linspace(0., 1., 3)
        mesh_vals = mesh**2
        # do not compare at right boundary because it will be zero
        interp_mesh = np.linspace(0., 1., num_mesh_points)[:-1]
        interp_vals = canonical_piecewise_quadratic_interpolation(
            interp_mesh, mesh_vals)
        assert np.allclose(interp_vals, interp_mesh**2)

    def test_piecewise_quadratic_interpolation(self):
        def function(x):
            return (x-0.5)**3
        num_mesh_points = 301
        mesh = np.linspace(0., 1., num_mesh_points)
        mesh_vals = function(mesh)
        # interp_mesh = np.random.uniform(0.,1.,101)
        interp_mesh = np.linspace(0., 1., 1001)
        ranges = [0, 1]
        interp_vals = piecewise_quadratic_interpolation(
            interp_mesh, mesh, mesh_vals, ranges)
        # print np.linalg.norm(interp_vals-function(interp_mesh))
        # import pylab as plt
        # I= np.argsort(interp_mesh)
        # plt.plot(interp_mesh[I],interp_vals[I],'k-')
        # plt.plot(mesh,mesh_vals,'o')
        # plt.show()
        assert np.linalg.norm(interp_vals-function(interp_mesh)) < 1e-6


if __name__ == '__main__':
    tensorprod_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestTensorProd)
    unittest.TextTestRunner(verbosity=2).run(tensorprod_test_suite)
    
