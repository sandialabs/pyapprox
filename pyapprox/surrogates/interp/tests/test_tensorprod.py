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
    piecewise_univariate_quadratic_quad_rule, irregular_piecewise_linear_basis,
    irregular_piecewise_quadratic_basis, irregular_piecewise_cubic_basis,
    get_univariate_interpolation_basis, TensorProductInterpolant
)


class TestTensorProd(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_irregular_piecewise_polynomial_basis(self):
        def fun(xx):
            return np.sum(xx**2, axis=0)[:, None]
        nnodes = 11
        lb, ub = 0, 1
        # creat nodes with random spacing
        nodes = np.linspace(lb, ub, nnodes*2)
        nodes = np.sort(np.hstack((nodes[[0, -1]], nodes[
            1+np.random.permutation(2*nnodes-2)[:nnodes-2]])))

        # check basis interpolates values at nodes
        samples = nodes
        values = fun(nodes[None, :])
        basis = irregular_piecewise_linear_basis(nodes, samples)
        assert np.allclose(basis @ values, fun(samples[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 41
        nodes = np.linspace(lb, ub, nnodes)
        samples = np.random.uniform(lb, ub, (nsamples))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_linear_basis(nodes, samples)
        # print(np.abs((basis @ values-fun(samples[None, :]))).max())
        # check basis interpolates values at nodes
        assert np.allclose(basis @ values, fun(samples[None, :]), atol=2e-4)

        def fun(xx):
            return np.sum(xx**3, axis=0)[:, None]
        nnodes = 3
        lb, ub = 0, 1
        # create nodes with random spacing
        nodes = np.linspace(lb, ub, nnodes*2)
        nodes = np.sort(np.hstack((nodes[[0, -1]], nodes[
           1+np.random.permutation(2*nnodes-2)[:nnodes-2]])))
        # nodes = np.linspace(lb, ub, nnodes)

        # check basis interpolates values at nodes
        samples = nodes
        values = fun(nodes[None, :])
        basis = irregular_piecewise_quadratic_basis(nodes, samples)
        assert np.allclose(basis @ values, fun(samples[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 41
        nodes = np.linspace(lb, ub, nnodes)
        samples = np.random.uniform(lb, ub, (nsamples))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_quadratic_basis(nodes, samples)
        # check basis interpolates values at nodes
        assert np.allclose(basis @ values, fun(samples[None, :]), atol=1e-5)

        nnodes = 10
        nodes = np.linspace(lb, ub, nnodes*2)
        nodes = np.sort(np.hstack((nodes[[0, -1]], nodes[
           1+np.random.permutation(2*nnodes-2)[:nnodes-2]])))
        basis = irregular_piecewise_cubic_basis(nodes, nodes)
        values = fun(nodes[None, :])
        assert np.allclose(basis @ values, fun(nodes[None, :]))

        # check basis accuracy is high with large nnodes
        nsamples = 31
        nnodes = 34
        nodes = np.linspace(lb, ub, nnodes)
        samples = np.random.uniform(lb, ub, (nsamples))
        values = fun(nodes[None, :])
        basis = irregular_piecewise_cubic_basis(nodes, samples)
        # check basis interpolates values at nodes
        assert np.allclose(basis @ values, fun(samples[None, :]), atol=1e-15)

        # import matplotlib.pyplot as plt
        # print(basis.max(axis=0))
        # plt.plot(samples, basis)
        # plt.plot(samples, fun(samples[None, :]))
        # plt.plot(samples, basis @ values, '--')
        # plt.plot(nodes, values, 'ko', ms=20)
        # plt.show()
        # assert np.allclose(basis @ values, fun(samples[None, :]))

    def _check_tensor_product_interpolation(self, basis_types, nnodes_1d, atol):
        nvars = len(basis_types)
        nnodes_1d = np.array(nnodes_1d)
        bases_1d = [
            get_univariate_interpolation_basis(bt) for bt in basis_types]
        interp = TensorProductInterpolant(bases_1d)
        nodes_1d = [np.linspace(0, 1, N) for N in nnodes_1d]

        def fun(samples):
            # when nnodes_1d is zero to test interpolation make sure
            # function is constant in that direction
            return np.sum(samples[nnodes_1d > 1]**3, axis=0)[:, None]

        train_samples = interp.tensor_product_grid(nodes_1d)
        train_values = fun(train_samples)
        interp.fit(nodes_1d, train_values)

        test_samples = np.random.uniform(0, 1, (nvars, 21))
        approx_values = interp(test_samples)
        test_values = fun(test_samples)
        assert np.allclose(test_values, approx_values, atol=atol)

    def test_tensor_product_interpolation(self):
        test_cases = [
            [["linear", "linear"], [41, 43], 1e-3],
            [["quadratic", "quadratic"], [41, 43], 1e-5],
            [["cubic", "cubic"], [40, 40], 1e-15],
            [["lagrange", "lagrange"], [4, 5], 1e-15],
            [["linear", "quadratic"], [41, 43], 1e-3],
            [["linear", "quadratic", "lagrange"], [41, 23, 4], 1e-3],
            [["cubic", "quadratic", "lagrange"], [25, 23, 4], 1e-4],
            # Following tests use of active vars when nnodes_1dii] = 0
            [["linear", "quadratic", "lagrange"], [1, 23, 4], 1e-4],
        ]
        for test_case in test_cases:
            self._check_tensor_product_interpolation(*test_case)

    def test_univariate_interpolant_quadrature(self):
        def fun(degree, xx):
            return np.sum(xx**degree, axis=0)[:, None]

        basis = get_univariate_interpolation_basis("linear")
        # randomize node spacing keeping both end points
        nnodes = 101
        nodes = np.linspace(-1, 1, 2*nnodes)
        nodes = np.sort(np.hstack((nodes[[0, -1]], nodes[
           1+np.random.permutation(2*nnodes-2)[:nnodes-2]])))
        weights = basis.quadrature_weights(nodes)
        assert np.allclose(fun(2, nodes[None, :]).T @ weights, 2/3, atol=1e-3)

        basis = get_univariate_interpolation_basis("quadratic")
        nodes = np.linspace(-1, 1, 3)
        weights = basis.quadrature_weights(nodes)
        assert np.allclose(fun(2, nodes[None, :]).T @ weights, 2/3, atol=1e-15)

        basis = get_univariate_interpolation_basis("quadratic")
        nodes = np.linspace(-1, 1, 33)
        weights = basis.quadrature_weights(nodes)
        assert np.allclose(fun(4, nodes[None, :]).T @ weights, 2/5, atol=1e-5)

        basis = get_univariate_interpolation_basis("cubic")
        nodes = np.linspace(-1, 1, 4)
        weights = basis.quadrature_weights(nodes)
        assert np.allclose(fun(3, nodes[None, :]).T @ weights, 0, atol=1e-15)

        basis = get_univariate_interpolation_basis("cubic")
        nodes = np.linspace(-1, 1, 34)
        weights = basis.quadrature_weights(nodes)
        assert np.allclose(fun(4, nodes[None, :]).T @ weights, 2/5, atol=1e-5)

    def test_get_tensor_product_piecewise_linear_quadrature_rule(self):
        nsamples_1d = 101

        def fun(xx):
            return np.sum(xx**2, axis=0)[:, None]

        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [-1, 1, -1, 1], 1)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 8/3.
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
        assert np.allclose(integral, true_integral, atol=1e-3)

        nsamples_1d = 3

        def fun(xx):
            return np.sum(xx**2, axis=0)[:, None]
        xx, ww = get_tensor_product_piecewise_polynomial_quadrature_rule(
            nsamples_1d, [-1, 1, -1, 1], 2)
        vals = fun(xx)
        integral = vals[:, 0].dot(ww)
        true_integral = 2/3*4
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
        assert np.allclose(interp_vals[:, 0], function(interp_mesh), atol=1e-5)

    def test_piecewise_poly_quadrature(self):
        npoints = 3
        range_1d = [0, 2]
        xx, ww = piecewise_univariate_linear_quad_rule(range_1d, npoints)
        assert np.allclose(xx, [0, 1, 2])
        vals = np.random.uniform(0, 1, npoints)
        a, b, c = vals
        true_integral = min(a, b)+min(b, c)+0.5*abs(b-a)+0.5*abs(c-b)
        assert np.allclose(true_integral, vals.dot(ww))

        npoints = 3
        range_1d = [0, 2]
        xx, ww = piecewise_univariate_quadratic_quad_rule(range_1d, npoints)
        assert np.allclose(xx, [0, 1, 2])
        coef = np.random.uniform(0, 1, npoints)
        vals = coef[0]+coef[1]*xx+coef[2]*xx**2
        true_integral = coef[0]*2+2**2*coef[1]/2+2**3*coef[2]/3
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
