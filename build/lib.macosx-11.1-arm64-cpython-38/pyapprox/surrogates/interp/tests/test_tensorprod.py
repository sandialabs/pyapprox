import numpy as np
import unittest
from functools import partial

from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.interp.tensorprod import (
    piecewise_quadratic_interpolation,
    canonical_piecewise_quadratic_interpolation,
    get_tensor_product_piecewise_polynomial_quadrature_rule,
    tensor_product_piecewise_polynomial_interpolation,
    piecewise_univariate_linear_quad_rule,
    piecewise_univariate_quadratic_quad_rule
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

        nsamples_1d = 3

        def fun(xx):
            return np.sum(xx**2, axis=0)[:, None]
        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [-1, 1, -1, 1], 2)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 2/3*4
        print(xx)
        print(ww)
        print(integral-true_integral)
        assert np.allclose(integral, true_integral, atol=1e-3)

        num_samples_1d = 101

        def fun(xx):
            return np.sum(xx**3, axis=0)[:, None]
        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [0, 1, 0, 2], 2)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 9/2
        assert np.allclose(integral, true_integral, atol=1e-3)

    def check_piecewise_poly_basis(self, basis_type, levels, tol):

        samples = np.random.uniform(-1, 1, (len(levels), 21))
        # print(samples.min(axis=1), samples.max(axis=1))
        # assert False

        def fun(samples):
            # when levels is zero to test interpolation make sure
            # function is constant in that direction
            return np.sum(samples[np.array(levels) > 0]**2, axis=0)[:, None]

        interp_fun = partial(tensor_product_piecewise_polynomial_interpolation,
                             levels=levels, fun=fun, basis_type=basis_type)
        # print(fun(samples).T)
        # print(interp_fun(samples).T)
        # print(((fun(samples)-interp_fun(samples))/fun(samples).T))
        # from pyapprox.util.visualization import get_meshgrid_function_data, plt
        # II = np.argsort(samples[0])
        # plt.plot(samples[0, II], fun(samples)[II], "-")
        # plt.plot(samples[0, II], interp_fun(samples)[II], "--")
        # plt.show()
        assert np.allclose(interp_fun(samples), fun(samples), rtol=tol)
        # X, Y, Z = get_meshgrid_function_data(
        #     lambda x: interp_fun(x)-fun(x), [0, 1, 0, 1], 50, qoi=0)
        # p = plt.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), 30))
        # plt.colorbar(p)
        # print(Z.max())
        # #plt.show()

    def test_piecewise_poly_basis(self):
        self.check_piecewise_poly_basis("quadratic", [1], 1e-8)
        self.check_piecewise_poly_basis("quadratic", [0, 1], 1e-8)
        self.check_piecewise_poly_basis("quadratic", [1, 1], 1e-8)
        self.check_piecewise_poly_basis("linear", [0, 10], 4e-4)
        self.check_piecewise_poly_basis("linear", [10, 10], 4e-4)

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
        assert np.allclose(interp_vals[:, 0], interp_mesh**2)

    def test_piecewise_quadratic_interpolation(self):
        def function(x):
            return (x-0.5)**3
        ranges = [0, 2]
        num_mesh_points = 101
        mesh = np.linspace(*ranges, num_mesh_points)
        mesh_vals = function(mesh)
        # interp_mesh = np.random.uniform(0.,1.,101)
        interp_mesh = np.linspace(*ranges, 1001)
        interp_vals = piecewise_quadratic_interpolation(
            interp_mesh, mesh, mesh_vals, ranges)
        # import pylab as plt
        # II = np.argsort(interp_mesh)
        # plt.plot(mesh, mesh_vals, 'o')
        # plt.plot(interp_mesh[II], interp_vals[II], 'k-')
        # plt.plot(interp_mesh, function(interp_mesh), 'r--')
        # plt.show()
        print(np.linalg.norm(interp_vals[:, 0]-function(interp_mesh)))
        assert np.allclose(interp_vals[:, 0], function(interp_mesh), atol=1e-5)

    def test_piecewise_poly_quadrature(self):
        npoints = 3
        range_1d = [0, 2]
        xx, ww = piecewise_univariate_linear_quad_rule(range_1d, npoints)
        assert np.allclose(xx, [0, 1, 2])
        vals = np.random.uniform(0, 1, npoints)
        a, b, c = vals
        true_integral = min(a, b)+min(b, c)+0.5*abs(b-a)+0.5*abs(c-b)
        print(true_integral, vals.dot(ww))
        assert np.allclose(true_integral, vals.dot(ww))

        npoints = 3
        range_1d = [0, 2]
        xx, ww = piecewise_univariate_quadratic_quad_rule(range_1d, npoints)
        assert np.allclose(xx, [0, 1, 2])
        coef = np.random.uniform(0, 1, npoints)
        vals = coef[0]+coef[1]*xx+coef[2]*xx**2
        true_integral = coef[0]*2+2**2*coef[1]/2+2**3*coef[2]/3
        print(true_integral, vals.dot(ww))
        assert np.allclose(true_integral, vals.dot(ww))

        npoints = 5
        range_1d = [0, 2]
        xx, ww = piecewise_univariate_quadratic_quad_rule(range_1d, npoints)
        mesh_vals = np.random.uniform(*range_1d, npoints)
        samples = np.random.uniform(*range_1d, int(1e6))
        vals = piecewise_quadratic_interpolation(
            samples, xx, mesh_vals, range_1d)
        assert np.allclose(vals.mean()*(range_1d[1]-range_1d[0]),
                           mesh_vals.dot(ww), atol=1e-3)


if __name__ == '__main__':
    tensorprod_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestTensorProd)
    unittest.TextTestRunner(verbosity=2).run(tensorprod_test_suite)
