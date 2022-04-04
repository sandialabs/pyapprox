import unittest
import numpy as np
from functools import partial

from pyapprox.utilities.utilities import (
    cartesian_product, outer_product, evaluate_quadratic_form,
    leave_many_out_lsq_cross_validation, leave_one_out_lsq_cross_validation,
    get_random_k_fold_sample_indices,
    get_cross_validation_rsquared_coefficient_of_variation,
    piecewise_quadratic_interpolation,
    get_tensor_product_piecewise_polynomial_quadrature_rule,
    integrate_using_univariate_gauss_legendre_quadrature_unbounded,
    canonical_piecewise_quadratic_interpolation,
    get_tensor_product_quadrature_rule,
    tensor_product_piecewise_polynomial_interpolation
    
)


class TestUtilities(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_cartesian_product(self):
        # test when num elems = 1
        s1 = np.arange(0, 3)
        s2 = np.arange(3, 5)

        sets = np.array([[0, 3], [1, 3], [2, 3], [0, 4],
                         [1, 4], [2, 4]], np.int)
        output_sets = cartesian_product([s1, s2], 1)
        assert np.array_equal(output_sets.T, sets)

        # # test when num elems > 1
        # s1 = np.arange( 0, 6 )
        # s2 = np.arange( 6, 10 )

        # sets = np.array( [[ 0, 1, 6, 7], [ 2, 3, 6, 7],
        #                   [ 4, 5, 6, 7], [ 0, 1, 8, 9],
        #                   [ 2, 3, 8, 9], [ 4, 5, 8, 9]], np.int )
        # output_sets = cartesian_product( [s1,s2], 2 )
        # assert np.array_equal( output_sets.T, sets )

    def test_outer_product(self):
        s1 = np.arange(0, 3)
        s2 = np.arange(3, 5)

        test_vals = np.array([0., 3., 6., 0., 4., 8.])
        output = outer_product([s1, s2])
        assert np.allclose(test_vals, output)

        output = outer_product([s1])
        assert np.allclose(output, s1)

    def test_evaluate_quadratic_form(self):
        nvars, nsamples = 3, 10
        A = np.random.normal(0, 1, nvars)
        A = A.T.dot(A)
        samples = np.random.uniform(0, 1, (nvars, nsamples))
        values1 = evaluate_quadratic_form(A, samples)

        values2 = np.zeros(samples.shape[1])
        for ii in range(samples.shape[1]):
            values2[ii] = samples[:, ii:ii+1].T.dot(A).dot(samples[:, ii:ii+1])

        assert np.allclose(values1, values2)

    def test_least_squares_loo_cross_validation(self):
        degree = 2
        alpha = 1e-3
        nsamples = 2*(degree+1)
        samples = np.random.uniform(-1, 1, (1, nsamples))
        basis_mat = samples.T**np.arange(degree+1)
        values = np.exp(samples).T
        cv_errors, cv_score, coef = leave_one_out_lsq_cross_validation(
            basis_mat, values, alpha)
        true_cv_errors = np.empty_like(cv_errors)
        for ii in range(nsamples):
            samples_ii = np.hstack((samples[:, :ii], samples[:, ii+1:]))
            basis_mat_ii = samples_ii.T**np.arange(degree+1)
            values_ii = np.vstack((values[:ii], values[ii+1:]))
            coef_ii = np.linalg.lstsq(
                basis_mat_ii.T.dot(basis_mat_ii) +
                alpha*np.eye(basis_mat.shape[1]
                             ), basis_mat_ii.T.dot(values_ii),
                rcond=None)[0]
            true_cv_errors[ii] = (basis_mat[ii].dot(coef_ii)-values[ii])
        assert np.allclose(cv_errors, true_cv_errors)
        assert np.allclose(
            cv_score, np.sqrt(np.sum(true_cv_errors**2, axis=0)/nsamples))

    def test_leave_many_out_lsq_cross_validation(self):
        degree = 2
        nsamples = 2*(degree+1)
        samples = np.random.uniform(-1, 1, (1, nsamples))
        basis_mat = samples.T**np.arange(degree+1)
        values = np.exp(samples).T*100
        alpha = 1e-3  # ridge regression regularization parameter value

        assert nsamples % 2 == 0
        nfolds = nsamples//3
        fold_sample_indices = get_random_k_fold_sample_indices(
            nsamples, nfolds)
        cv_errors, cv_score, coef = leave_many_out_lsq_cross_validation(
            basis_mat, values, fold_sample_indices, alpha)

        true_cv_errors = np.empty_like(cv_errors)
        for kk in range(len(fold_sample_indices)):
            K = np.ones(nsamples, dtype=bool)
            K[fold_sample_indices[kk]] = False
            basis_mat_kk = basis_mat[K, :]
            gram_mat_kk = basis_mat_kk.T.dot(basis_mat_kk) + np.eye(
                basis_mat_kk.shape[1])*alpha
            values_kk = basis_mat_kk.T.dot(values[K, :])
            coef_kk = np.linalg.lstsq(gram_mat_kk, values_kk, rcond=None)[0]
            true_cv_errors[kk] = basis_mat[fold_sample_indices[kk], :].dot(
                coef_kk)-values[fold_sample_indices[kk]]
        # print(cv_errors, true_cv_errors)
        assert np.allclose(cv_errors, true_cv_errors)
        true_cv_score = np.sqrt((true_cv_errors**2).sum(axis=(0, 1))/nsamples)
        assert np.allclose(true_cv_score, cv_score)

        rsq = get_cross_validation_rsquared_coefficient_of_variation(
            cv_score, values)

        print(rsq)

    
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
        # from pyapprox.utilities.visualization import plt, get_meshgrid_function_data
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

    
    def test_integrate_using_univariate_gauss_legendre_quadrature_unbounded(self):
        from scipy.stats import norm, gamma, beta

        # unbounded
        rv = norm(0, 1)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # left bounded
        rv = gamma(1)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # bounded
        rv = beta(20, 10, -2, 5)

        def integrand(x):
            return rv.pdf(x)[:, None]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, 1)

        # multiple qoi
        rv = norm(2, 3)

        def integrand(x):
            return rv.pdf(x)[:, None]*x[:, None]**np.arange(3)[None, :]
        lb, ub = rv.interval(1)
        nquad_samples = 100
        res = integrate_using_univariate_gauss_legendre_quadrature_unbounded(
            integrand, lb, ub, nquad_samples, interval_size=2)
        assert np.allclose(res, [1, 2, 3**2+2**2])

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



if __name__ == "__main__":
    utilities_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestUtilities)
    unittest.TextTestRunner(verbosity=2).run(utilities_test_suite)
