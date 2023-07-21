import unittest
from scipy import special as sp
import numpy as np
from functools import partial
from scipy import stats
from scipy.stats._continuous_distns import rv_continuous

from pyapprox.util.utilities import (
    approx_jacobian, approx_fprime, cartesian_product, outer_product,
    get_all_sample_combinations,
    integrate_using_univariate_gauss_legendre_quadrature_unbounded,
    get_tensor_product_quadrature_rule
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, evaluate_multivariate_orthonormal_polynomial,
    define_poly_options_from_variable_transformation,
    get_polynomial_from_variable,
    compute_product_coeffs_1d_for_each_variable,
    multiply_multivariate_orthonormal_polynomial_expansions,
    compute_multivariate_orthonormal_basis_product,
    compute_univariate_orthonormal_basis_products,
    conditional_moments_of_polynomial_chaos_expansion,
    get_univariate_quadrature_rules_from_variable
)
from pyapprox.surrogates.interp.indexing import (
    compute_hyperbolic_indices, tensor_product_indices,
    sort_indices_lexiographically
)
from pyapprox.surrogates.orthopoly.quadrature import (
    gauss_hermite_pts_wts_1D, gauss_jacobi_pts_wts_1D
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation,
    AffineTransform
)
from pyapprox.variables.marginals import (
    float_rv_discrete, rv_function_indpndt_vars, rv_product_indpndt_vars
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d
)
from pyapprox.optimization.quantile_regression import (
    solve_least_squares_regression
)


class TestGPC(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_evaluate_multivariate_orthonormal_polynomial(self):
        num_vars = 2
        alpha = 0.
        beta = 0.
        degree = 2
        deriv_order = 1
        probability_measure = True

        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=probability_measure)

        x, w = np.polynomial.legendre.leggauss(degree)
        samples = cartesian_product([x]*num_vars, 1)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        # sort lexographically to make testing easier
        II = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, II]

        basis_matrix = evaluate_multivariate_orthonormal_polynomial(
            samples, indices, ab, deriv_order)

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x, x, 0.5*(3.*x**2-1)]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x, 1.0+0.*x, 3.*x]).T)
            exact_basis_vals_1d[-1] /= np.sqrt(1./(2*np.arange(degree+1)+1))
            exact_basis_derivs_1d[-1] /= np.sqrt(1./(2*np.arange(degree+1)+1))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:, 0], exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1], exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, exact_basis_derivs_1d[0][:, 1], 0.*x,
             exact_basis_derivs_1d[0][:, 2],
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, 0.*x, exact_basis_derivs_1d[1][:, 1], 0.*x,
             exact_basis_vals_1d[0][:, 1]*exact_basis_derivs_1d[1][:, 1],
             exact_basis_derivs_1d[1][:, 2]]).T))

        def func(x): return evaluate_multivariate_orthonormal_polynomial(
            x, indices, ab, 0)
        basis_matrix_derivs = basis_matrix[samples.shape[1]:]
        basis_matrix_derivs_fd = np.empty_like(basis_matrix_derivs)
        for ii in range(samples.shape[1]):
            basis_matrix_derivs_fd[ii::samples.shape[1], :] = approx_fprime(
                samples[:, ii:ii+1], func, 1e-7)
        assert np.allclose(
            exact_basis_matrix[samples.shape[1]:], basis_matrix_derivs_fd)

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_jacobi_pce(self):
        num_vars = 2
        degree = 2

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        samples, weights = get_tensor_product_quadrature_rule(
            degree-1, num_vars, np.polynomial.legendre.leggauss)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        # sort lexographically to make testing easier
        II = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, II]
        # remove [0,2] index so max_level is not the same for every dimension
        # also remove [1,0] and [1,1] to make sure can handle index sets that
        # have missing univariate degrees not at the ends
        J = [1, 5, 4]
        reduced_indices = np.delete(indices, J, axis=1)
        poly.set_indices(reduced_indices)

        basis_matrix = poly.basis_matrix(samples, {'deriv_order': 1})

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x, x, 0.5*(3.*x**2-1)]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x, 1.0+0.*x, 3.*x]).T)
            exact_basis_vals_1d[-1] /= np.sqrt(1./(2*np.arange(degree+1)+1))
            exact_basis_derivs_1d[-1] /= np.sqrt(1./(2*np.arange(degree+1)+1))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:, 0], exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1], exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, exact_basis_derivs_1d[0][:, 1], 0.*x,
             exact_basis_derivs_1d[0][:, 2],
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, 0.*x, exact_basis_derivs_1d[1][:, 1], 0.*x,
             exact_basis_vals_1d[0][:, 1]*exact_basis_derivs_1d[1][:, 1],
             exact_basis_derivs_1d[1][:, 2]]).T))

        exact_basis_matrix = np.delete(exact_basis_matrix, J, axis=1)

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_hermite_pce(self):
        num_vars = 2
        degree = 2

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.norm(0, 1), num_vars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, gauss_hermite_pts_wts_1D)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        # sort lexographically to make testing easier
        II = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, II]
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples, {'deriv_order': 1})

        vals_basis_matrix = basis_matrix[:samples.shape[1], :]
        inner_products = (vals_basis_matrix.T*weights).dot(vals_basis_matrix)
        assert np.allclose(inner_products, np.eye(basis_matrix.shape[1]))

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x, x, x**2-1]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x, 1.0+0.*x, 2.*x]).T)
            exact_basis_vals_1d[-1] /= np.sqrt(
                sp.factorial(np.arange(degree+1)))
            exact_basis_derivs_1d[-1] /= np.sqrt(
                sp.factorial(np.arange(degree+1)))

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:, 0], exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1], exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, exact_basis_derivs_1d[0][:, 1], 0.*x,
             exact_basis_derivs_1d[0][:, 2],
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, 0.*x, exact_basis_derivs_1d[1][:, 1], 0.*x,
             exact_basis_vals_1d[0][:, 1]*exact_basis_derivs_1d[1][:, 1],
             exact_basis_derivs_1d[1][:, 2]]).T))

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_mixed_basis_pce(self):
        degree = 2

        gauss_mean, gauss_var = -1, 4
        univariate_variables = [
            stats.uniform(-1, 2), stats.norm(gauss_mean, np.sqrt(gauss_var)),
            stats.uniform(0, 3)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        univariate_quadrature_rules = [
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=0, beta_poly=0),
            gauss_hermite_pts_wts_1D,
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=0, beta_poly=0)]
        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, univariate_quadrature_rules,
            var_trans.map_from_canonical)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        # sort lexographically to make testing easier
        indices = sort_indices_lexiographically(indices)
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples, {'deriv_order': 1})
        vals_basis_matrix = basis_matrix[:samples.shape[1], :]
        inner_products = (vals_basis_matrix.T*weights).dot(vals_basis_matrix)
        assert np.allclose(inner_products, np.eye(basis_matrix.shape[1]))

        samples = variable.rvs(3)
        basis_matrix = poly.basis_matrix(samples, {'deriv_order': 1})
        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd, :].copy()
            if dd == 0 or dd == 2:
                if dd == 2:
                    # y = x/3
                    # z = 2*y-1=2*x/3-1=2/3*x-3/2*2/3=2/3*(x-3/2)=(x-3/2)/(3/2)
                    loc, scale = 3/2, 3/2
                    x = (x-loc)/scale
                exact_basis_vals_1d.append(
                    np.asarray([1+0.*x, x, 0.5*(3.*x**2-1)]).T)
                exact_basis_derivs_1d.append(
                    np.asarray([0.*x, 1.0+0.*x, 3.*x]).T)
                exact_basis_vals_1d[-1] /= np.sqrt(1. /
                                                   (2*np.arange(degree+1)+1))
                exact_basis_derivs_1d[-1] /= np.sqrt(
                    1./(2*np.arange(degree+1)+1))
                # account for affine transformation in derivs
                if dd == 2:
                    exact_basis_derivs_1d[-1] /= scale
            elif dd == 1:
                loc, scale = gauss_mean, np.sqrt(gauss_var)
                x = (x-loc)/scale
                exact_basis_vals_1d.append(
                    np.asarray([1+0.*x, x, x**2-1]).T)
                exact_basis_derivs_1d.append(
                    np.asarray([0.*x, 1.0+0.*x, 2.*x]).T)
                exact_basis_vals_1d[-1] /= np.sqrt(
                    sp.factorial(np.arange(degree+1)))
                exact_basis_derivs_1d[-1] /= np.sqrt(
                    sp.factorial(np.arange(degree+1)))
                # account for affine transformation in derivs
                exact_basis_derivs_1d[-1] /= scale

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:, 0],
             exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[2][:, 1],
             exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[2][:, 1],
             exact_basis_vals_1d[1][:, 1]*exact_basis_vals_1d[2][:, 1],
             exact_basis_vals_1d[2][:, 2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x,
             exact_basis_derivs_1d[0][:, 1],
             0.*x,
             0*x,
             exact_basis_derivs_1d[0][:, 2],
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             0.*x,
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[2][:, 1],
             0.*x,
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x,
             0.*x,
             exact_basis_derivs_1d[1][:, 1],
             0.*x,
             0*x,
             exact_basis_derivs_1d[1][:, 1]*exact_basis_vals_1d[0][:, 1],
             exact_basis_derivs_1d[1][:, 2],
             0.*x,
             exact_basis_derivs_1d[1][:, 1]*exact_basis_vals_1d[2][:, 1],
             0.*x]).T))

        # x3 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x,
             0.*x,
             0.*x,
             exact_basis_derivs_1d[2][:, 1],
             0*x,
             0*x,
             0*x,
             exact_basis_derivs_1d[2][:, 1]*exact_basis_vals_1d[0][:, 1],
             exact_basis_derivs_1d[2][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_derivs_1d[2][:, 2]]).T))

        func = poly.basis_matrix
        exact_basis_matrix_derivs = exact_basis_matrix[samples.shape[1]:]
        basis_matrix_derivs_fd = np.empty_like(exact_basis_matrix_derivs)
        for ii in range(samples.shape[1]):
            basis_matrix_derivs_fd[ii::samples.shape[1], :] = approx_fprime(
                samples[:, ii:ii+1], func)

        # print(np.linalg.norm(
        #    exact_basis_matrix_derivs-basis_matrix_derivs_fd,
        #    ord=np.inf))
        # print(exact_basis_matrix_derivs)
        # print(basis_matrix_derivs_fd)
        assert np.allclose(
            exact_basis_matrix_derivs, basis_matrix_derivs_fd,
            atol=1e-7, rtol=1e-7)
        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_monomial_pce(self):
        num_vars = 2
        degree = 2

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            rv_continuous(name="continuous_monomial")(), num_vars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        def univariate_quadrature_rule(nn):
            x, w = gauss_jacobi_pts_wts_1D(nn, 0, 0)
            x = (x+1)/2.
            return x, w

        samples, weights = get_tensor_product_quadrature_rule(
            degree, num_vars, univariate_quadrature_rule)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        # sort lexographically to make testing easier
        II = np.lexsort((indices[0, :], indices[1, :], indices.sum(axis=0)))
        indices = indices[:, II]
        poly.set_indices(indices)

        basis_matrix = poly.basis_matrix(samples, {'deriv_order': 1})

        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(num_vars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                np.asarray([1+0.*x, x, x**2]).T)
            exact_basis_derivs_1d.append(np.asarray([0.*x, 1.0+0.*x, 2.*x]).T)

        exact_basis_matrix = np.asarray(
            [exact_basis_vals_1d[0][:, 0], exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1], exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2]]).T

        # x1 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, exact_basis_derivs_1d[0][:, 1], 0.*x,
             exact_basis_derivs_1d[0][:, 2],
             exact_basis_derivs_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             0.*x]).T))

        # x2 derivative
        exact_basis_matrix = np.vstack((exact_basis_matrix, np.asarray(
            [0.*x, 0.*x, exact_basis_derivs_1d[1][:, 1], 0.*x,
             exact_basis_vals_1d[0][:, 1]*exact_basis_derivs_1d[1][:, 1],
             exact_basis_derivs_1d[1][:, 2]]).T))

        assert np.allclose(exact_basis_matrix, basis_matrix)

    def test_evaluate_multivariate_mixed_basis_pce_moments(self):
        degree = 2

        alpha_stat, beta_stat = 2, 3
        univariate_variables = [
            stats.beta(alpha_stat, beta_stat, 0, 1), stats.norm(-1, 2)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        univariate_quadrature_rules = [
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=beta_stat-1,
                    beta_poly=alpha_stat-1), gauss_hermite_pts_wts_1D]
        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, univariate_quadrature_rules,
            var_trans.map_from_canonical)

        coef = np.ones((indices.shape[1], 2))
        coef[:, 1] *= 2
        poly.set_coefficients(coef)
        basis_matrix = poly.basis_matrix(samples)
        values = basis_matrix.dot(coef)
        true_mean = values.T.dot(weights)
        true_variance = (values.T**2).dot(weights)-true_mean**2

        assert np.allclose(poly.mean(), true_mean)
        assert np.allclose(poly.variance(), true_variance)

        assert np.allclose(np.diag(poly.covariance()), poly.variance())
        assert np.allclose(poly.covariance()[
                           0, 1], coef[1:, 0].dot(coef[1:, 1]))

    def test_pce_jacobian(self):
        degree = 2

        alpha_stat, beta_stat = 2, 3
        univariate_variables = [
            stats.beta(alpha_stat, beta_stat, 0, 1), stats.norm(-1, 2)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)
        num_vars = len(univariate_variables)

        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.set_indices(indices)

        sample = generate_independent_random_samples(variable, 1)

        coef = np.ones((indices.shape[1], 2))
        coef[:, 1] *= 2
        poly.set_coefficients(coef)

        jac = poly.jacobian(sample)
        fd_jac = approx_jacobian(
            lambda x: poly(x[:, np.newaxis])[0, :], sample[:, 0])
        assert np.allclose(jac, fd_jac)

    def test_hahn_hypergeometric(self):
        degree = 4
        M, n, N = 20, 7, 12
        rv = stats.hypergeom(M, n, N)
        var_trans = AffineTransform([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis, :])
        xk = np.arange(0, n+1)[np.newaxis, :].astype(float)
        p = poly.basis_matrix(xk)
        w = rv.pmf(xk[0, :])
        print(w)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_krawtchouk_binomial(self):
        degree = 4
        n, p = 10, 0.5
        rv = stats.binom(n, p)
        var_trans = AffineTransform([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis, :])
        xk = np.arange(0, n+1)[np.newaxis, :]
        p = poly.basis_matrix(xk)
        w = rv.pmf(xk[0, :])
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_discrete_chebyshev(self):
        N, degree = 10, 5
        xk, pk = np.arange(N).astype(float), np.ones(N)/N
        rv = float_rv_discrete(name='discrete_chebyshev', values=(xk, pk))()
        var_trans = AffineTransform([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis, :])
        p = poly.basis_matrix(xk[np.newaxis, :])
        w = pk
        # print((np.dot(p.T*w, p), np.eye(degree+1)))
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_float_rv_discrete_chebyshev(self):
        N, degree = 10, 5
        xk, pk = np.geomspace(1.0, 512.0, num=N), np.ones(N)/N
        rv = float_rv_discrete(name='float_rv_discrete', values=(xk, pk))()
        var_trans = AffineTransform([rv])
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly_opts['numerically_generated_poly_accuracy_tolerance'] = 1e-9
        poly.configure(poly_opts)
        poly.set_indices(np.arange(degree+1)[np.newaxis, :])
        p = poly.basis_matrix(xk[np.newaxis, :])
        w = pk
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_pce_for_gumbel_variable(self):
        degree = 3
        mean, std = 1e4, 7.5e3
        beta = std*np.sqrt(6)/np.pi
        mu = mean - beta*np.euler_gamma
        rv1 = stats.gumbel_r(loc=mu, scale=beta)
        assert np.allclose(rv1.mean(), mean) and np.allclose(rv1.std(), std)
        rv2 = stats.lognorm(1)
        for rv in [rv2, rv1]:
            var_trans = AffineTransform([rv])
            poly = PolynomialChaosExpansion()
            poly_opts = define_poly_options_from_variable_transformation(
                var_trans)
            poly_opts['numerically_generated_poly_accuracy_tolerance'] = 1e-9
            poly.configure(poly_opts)
            poly.set_indices(np.arange(degree+1)[np.newaxis, :])
            poly.set_coefficients(np.ones((poly.indices.shape[1], 1)))

            def integrand(x):
                p = poly.basis_matrix(x[np.newaxis, :])
                G = np.empty((x.shape[0], p.shape[1]**2))
                kk = 0
                for ii in range(p.shape[1]):
                    for jj in range(p.shape[1]):
                        G[:, kk] = p[:, ii]*p[:, jj]
                        kk += 1
                return G*rv.pdf(x)[:, None]

            lb, ub = rv.interval(1)
            interval_size = rv.interval(0.99)[1]-rv.interval(0.99)[0]
            interval_size *= 10
            res = \
                integrate_using_univariate_gauss_legendre_quadrature_unbounded(
                    integrand, lb, ub, 10, interval_size=interval_size,
                    verbose=0, max_steps=10000)
            res = np.reshape(
                res, (poly.indices.shape[1], poly.indices.shape[1]), order='C')
            # print('r', res-np.eye(degree+1))
            assert np.allclose(res, np.eye(degree+1), atol=1e-6)

    def test_conditional_moments_of_polynomial_chaos_expansion(self):
        num_vars = 3
        degree = 2
        inactive_idx = [0, 2]
        np.random.seed(1)
        # keep variables on canonical domain to make constructing
        # tensor product quadrature rule, used for testing, easier
        var = [stats.uniform(-1, 2), stats.beta(2, 2, -1, 2), stats.norm(0, 1)]
        quad_rules = [
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=0, beta_poly=0),
            partial(gauss_jacobi_pts_wts_1D, alpha_poly=1, beta_poly=1),
            partial(gauss_hermite_pts_wts_1D)]
        var_trans = AffineTransform(var)
        poly = PolynomialChaosExpansion()
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(compute_hyperbolic_indices(num_vars, degree, 1.0))
        poly.set_coefficients(
            np.arange(poly.indices.shape[1], dtype=float)[:, np.newaxis])

        fixed_samples = np.array(
            [[vv.rvs() for vv in np.array(var)[inactive_idx]]]).T
        mean, variance = conditional_moments_of_polynomial_chaos_expansion(
            poly, fixed_samples, inactive_idx, True)

        active_idx = np.setdiff1d(np.arange(num_vars), inactive_idx)
        random_samples, weights = get_tensor_product_quadrature_rule(
            [2*degree]*len(active_idx), len(active_idx),
            [quad_rules[ii] for ii in range(num_vars) if ii in active_idx])
        samples = get_all_sample_combinations(fixed_samples, random_samples)
        temp = samples[len(inactive_idx):].copy()
        samples[inactive_idx] = samples[:len(inactive_idx)]
        samples[active_idx] = temp

        true_mean = (poly(samples).T.dot(weights).T)
        true_variance = ((poly(samples)**2).T.dot(weights).T)-true_mean**2
        assert np.allclose(true_mean, mean)
        assert np.allclose(true_variance, variance)

        # nsamples = int(1e5)
        # samples = generate_independent_random_samples(
        #     var_trans.variable,nsamples)
        # samples[inactive_idx,:]=fixed_samples
        # values = poly(samples)
        # print(mean,values.mean())
        # print(variance,np.var(values))

    def test_compute_univariate_orthonormal_basis_products(self):
        max_degree1, max_degree2 = 3, 2

        get_recursion_coefficients = partial(
            jacobi_recurrence, alpha=0., beta=0., probability=True)

        product_coefs = compute_univariate_orthonormal_basis_products(
            get_recursion_coefficients, max_degree1, max_degree2)

        max_degree = max_degree1+max_degree2
        x = np.linspace(-1, 1, 51)
        recursion_coefs = get_recursion_coefficients(max_degree+1)
        ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(
            x, max_degree, recursion_coefs)

        kk = 0
        for d1 in range(max_degree1+1):
            for d2 in range(min(d1+1, max_degree2+1)):
                exact_product = \
                    ortho_basis_matrix[:, d1]*ortho_basis_matrix[:, d2]

                product = np.dot(
                    ortho_basis_matrix[:, :product_coefs[kk].shape[0]],
                    product_coefs[kk]).sum(axis=1)
                assert np.allclose(product, exact_product)
                kk += 1

    def test_compute_multivariate_orthonormal_basis_product(self):
        univariate_variables = [stats.norm(), stats.uniform()]
        variable = IndependentMarginalsVariable(
            univariate_variables)

        poly1 = get_polynomial_from_variable(variable)
        poly2 = get_polynomial_from_variable(variable)

        max_degrees1, max_degrees2 = [3, 3], [2, 2]
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
            poly1, max_degrees1, max_degrees2)

        for ii in range(max_degrees1[0]):
            for jj in range(max_degrees1[1]):
                poly_index_ii, poly_index_jj = np.array(
                    [ii, jj]), np.array([ii, jj])

                poly1.set_indices(poly_index_ii[:, np.newaxis])
                poly1.set_coefficients(np.ones([1, 1]))
                poly2.set_indices(poly_index_jj[:, np.newaxis])
                poly2.set_coefficients(np.ones([1, 1]))

                product_indices, product_coefs = \
                    compute_multivariate_orthonormal_basis_product(
                        product_coefs_1d, poly_index_ii, poly_index_jj,
                        max_degrees1, max_degrees2)

                poly_prod = get_polynomial_from_variable(variable)
                poly_prod.set_indices(product_indices)
                poly_prod.set_coefficients(product_coefs)

                samples = generate_independent_random_samples(variable, 5)
                # print(poly_prod(samples),poly1(samples)*poly2(samples))
            assert np.allclose(poly_prod(samples),
                               poly1(samples)*poly2(samples))

    def test_multiply_multivariate_orthonormal_polynomial_expansions(self):
        univariate_variables = [stats.norm(), stats.uniform()]
        variable = IndependentMarginalsVariable(
            univariate_variables)

        degree1, degree2 = 3, 2
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree1))
        poly1.set_coefficients(np.random.normal(
            0, 1, (poly1.indices.shape[1], 1)))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree2))
        poly2.set_coefficients(np.random.normal(
            0, 1, (poly2.indices.shape[1], 1)))

        max_degrees1 = poly1.indices.max(axis=1)
        max_degrees2 = poly2.indices.max(axis=1)
        product_coefs_1d = compute_product_coeffs_1d_for_each_variable(
            poly1, max_degrees1, max_degrees2)

        indices, coefs = \
            multiply_multivariate_orthonormal_polynomial_expansions(
                product_coefs_1d, poly1.get_indices(),
                poly1.get_coefficients(), poly2.get_indices(),
                poly2.get_coefficients())

        poly3 = get_polynomial_from_variable(variable)
        poly3.set_indices(indices)
        poly3.set_coefficients(coefs)

        samples = generate_independent_random_samples(variable, 10)
        # print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly3(samples), poly1(samples)*poly2(samples))

    def test_multiply_pce(self):
        np.random.seed(1)
        np.set_printoptions(precision=16)
        univariate_variables = [stats.norm(), stats.uniform()]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        degree1, degree2 = 1, 2
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree1))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree2))

        # coef1 = np.random.normal(0,1,(poly1.indices.shape[1],1))
        # coef2 = np.random.normal(0,1,(poly2.indices.shape[1],1))
        coef1 = np.arange(poly1.indices.shape[1])[:, np.newaxis]
        coef2 = np.arange(poly2.indices.shape[1])[:, np.newaxis]
        poly1.set_coefficients(coef1)
        poly2.set_coefficients(coef2)

        poly3 = poly1*poly2
        samples = generate_independent_random_samples(variable, 10)
        assert np.allclose(poly3(samples), poly1(samples)*poly2(samples))

        for order in range(4):
            poly = poly1**order
            assert np.allclose(poly(samples), poly1(samples)**order)

    def test_add_pce(self):
        univariate_variables = [stats.norm(), stats.uniform()]
        variable = IndependentMarginalsVariable(
            univariate_variables)
        degree1, degree2 = 2, 3
        poly1 = get_polynomial_from_variable(variable)
        poly1.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree1))
        poly1.set_coefficients(np.random.normal(
            0, 1, (poly1.indices.shape[1], 1)))
        poly2 = get_polynomial_from_variable(variable)
        poly2.set_indices(compute_hyperbolic_indices(
            variable.num_vars(), degree2))
        poly2.set_coefficients(np.random.normal(
            0, 1, (poly2.indices.shape[1], 1)))

        poly3 = poly1+poly2+poly2
        samples = generate_independent_random_samples(variable, 10)
        # print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly3(samples), poly1(samples)+2*poly2(samples))

        poly4 = poly1-poly2
        samples = generate_independent_random_samples(variable, 10)
        # print(poly3(samples),poly1(samples)*poly2(samples))
        assert np.allclose(poly4(samples), poly1(samples)-poly2(samples))

    def test_composition_of_orthonormal_polynomials(self):
        def fn1(z):
            # return W_1
            return (z[0, :]+3*z[0, :]**2)[:, None]

        def fn2(z):
            # return W_2
            return (1+z[0, :]*z[1, :])[:, None]

        def fn3(z):
            # z is just random variables
            return z[0:1, :].T + 35*(3*fn1(z)**2-1) + 3*z[0:1, :].T*fn2(z)

        def fn3_trans(x):
            """
            x is z_1, W_1, W_2
            """
            return (x[0:1, :] + 35*(3*x[1:2, :]**2-1) + 3*x[0:1, :]*x[2:3, :])

        nvars = 2
        samples = np.random.uniform(-1, 1, (nvars, 100))
        values = fn3(samples)

        indices = compute_hyperbolic_indices(nvars, 4, 1)
        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), nvars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(indices)
        basis_mat = poly.basis_matrix(samples)
        coef = np.linalg.lstsq(basis_mat, values, rcond=None)[0]
        mean = coef[0]
        variance = np.sum(coef[1:]**2)
        # print(mean, variance, 2595584/15-mean**2, 2059769/15)
        assert np.allclose(mean, 189)
        assert np.allclose(variance,  2595584/15-mean**2)

        samples = np.random.uniform(-1, 1, (nvars, 100000))
        basis_mat = poly.basis_matrix(samples)
        x = samples[0:1, :].T
        y = samples[1:2, :].T

        assert np.allclose(
            basis_mat.dot(coef), -35+4*x+3*x**2*y+105*x**2+630*x**3+945*x**4)
        assert np.allclose(2/np.sqrt(5)*basis_mat[:, 3:4], (3*x**2-1))
        assert np.allclose(
            basis_mat.dot(coef),
            4*x+3*x**2*y+2/np.sqrt(5)*35*basis_mat[:, 3:4]+630*x**3+945*x**4)
        assert np.allclose(
            basis_mat.dot(coef),
            382*x+3*x**2*y+2/np.sqrt(5)*35*basis_mat[:, 3:4] +
            2/np.sqrt(7)*126*basis_mat[:, 6:7]+945*x**4)
        assert np.allclose(
            basis_mat.dot(coef),
            382*x+3*x**2*y+2/np.sqrt(5)*35*basis_mat[:, 3:4] +
            2/np.sqrt(7)*126*basis_mat[:, 6:7] +
            8/np.sqrt(9)*27*basis_mat[:, 10:11]+810*x**2-81)
        assert np.allclose(
            basis_mat.dot(coef),
            -81+270+382*x+3*x**2*y+2/np.sqrt(5)*305*basis_mat[:, 3:4] +
            2/np.sqrt(7)*126*basis_mat[:, 6:7] +
            8/np.sqrt(9)*27*basis_mat[:, 10:11])
        assert np.allclose(
            basis_mat.dot(coef), 189+382*x+2/np.sqrt(5)*305*basis_mat[:, 3:4] +
            2/np.sqrt(7)*126*basis_mat[:, 6:7] +
            8/np.sqrt(9)*27*basis_mat[:, 10:11] +
            2/np.sqrt(15)*basis_mat[:, 8:9]+y)

        assert np.allclose(
            basis_mat.dot(coef),
            189 +
            1/np.sqrt(3) * 382*basis_mat[:, 1:2] +
            2/np.sqrt(5) * 305*basis_mat[:, 3:4] +
            2/np.sqrt(7) * 126*basis_mat[:, 6:7] +
            8/np.sqrt(9) * 27.*basis_mat[:, 10:11] +
            2/np.sqrt(15)*1.0*basis_mat[:, 8:9] +
            1/np.sqrt(3) * 1.0*basis_mat[:, 2:3])

        assert np.allclose(variance, 382**2/3+(2*305)**2 /
                           5+(2*126)**2/7+(8*27)**2/9+4/15+1/3)

    def test_pce_product_of_beta_variables(self):
        def fun(x):
            return np.sqrt(x.prod(axis=0))[:, None]

        dist_alpha1, dist_beta1 = 1, 1
        dist_alpha2, dist_beta2 = dist_alpha1+0.5, dist_beta1
        nvars = 2

        x_1d, w_1d = [], []
        nquad_samples_1d = 100
        x, w = gauss_jacobi_pts_wts_1D(
            nquad_samples_1d, dist_beta1-1, dist_alpha1-1)
        x = (x+1)/2
        x_1d.append(x)
        w_1d.append(w)
        x, w = gauss_jacobi_pts_wts_1D(
            nquad_samples_1d, dist_beta2-1, dist_alpha2-1)
        x = (x+1)/2
        x_1d.append(x)
        w_1d.append(w)

        quad_samples = cartesian_product(x_1d)
        quad_weights = outer_product(w_1d)

        mean = fun(quad_samples)[:, 0].dot(quad_weights)
        variance = (fun(quad_samples)[:, 0]**2).dot(quad_weights)-mean**2
        assert np.allclose(
            mean, stats.beta(dist_alpha1*2, dist_beta1*2).mean())
        assert np.allclose(
            variance, stats.beta(dist_alpha1*2, dist_beta1*2).var())

        degree = 10
        poly = PolynomialChaosExpansion()
        # the distribution and ranges of univariate variables is ignored
        # when var_trans.set_identity_maps([0]) is used
        initial_variables = [stats.uniform(0, 1)]
        # TODO get quad rules from initial variables
        quad_rules = [(x, w) for x, w in zip(x_1d, w_1d)]
        univariate_variables = [
            rv_function_indpndt_vars(fun, initial_variables, quad_rules)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(tensor_product_indices([degree]))

        train_samples = (np.linspace(0, np.pi, 101)[None, :]+1)/2
        train_vals = train_samples.T
        coef = np.linalg.lstsq(
            poly.basis_matrix(train_samples), train_vals, rcond=None)[0]
        poly.set_coefficients(coef)
        assert np.allclose(
            poly.mean(), stats.beta(dist_alpha1*2, dist_beta1*2).mean())
        assert np.allclose(
            poly.variance(), stats.beta(dist_alpha1*2, dist_beta1*2).var())

        poly = PolynomialChaosExpansion()
        initial_variables = [stats.uniform(0, 1)]
        funs = [lambda x: np.sqrt(x)]*nvars
        quad_rules = [(x, w) for x, w in zip(x_1d, w_1d)]
        # TODO get quad rules from initial variables
        univariate_variables = [
            rv_product_indpndt_vars(funs, initial_variables, quad_rules)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        poly.configure(poly_opts)
        poly.set_indices(tensor_product_indices([degree]))

        train_samples = (np.linspace(0, np.pi, 101)[None, :]+1)/2
        train_vals = train_samples.T
        coef = np.linalg.lstsq(
            poly.basis_matrix(train_samples), train_vals, rcond=None)[0]
        poly.set_coefficients(coef)
        assert np.allclose(
            poly.mean(), stats.beta(dist_alpha1*2, dist_beta1*2).mean())
        assert np.allclose(
            poly.variance(), stats.beta(dist_alpha1*2, dist_beta1*2).var())

    def test_hermite_basis_for_lognormal_variables(self):
        def function(x): return (x.T)**2

        degree = 2
        # mu_g, sigma_g = 1e1, 0.1
        mu_l, sigma_l = 2.1e11, 2.1e10
        mu_g = np.log(mu_l**2/np.sqrt(mu_l**2+sigma_l**2))
        sigma_g = np.sqrt(np.log(1+sigma_l**2/mu_l**2))

        lognorm = stats.lognorm(s=sigma_g, scale=np.exp(mu_g))
        # assert np.allclose([lognorm.mean(), lognorm.std()], [mu_l, sigma_l])

        univariate_variables = [stats.norm(mu_g, sigma_g)]
        var_trans = AffineTransform(univariate_variables)
        pce = PolynomialChaosExpansion()
        pce_opts = define_poly_options_from_variable_transformation(
            var_trans)
        pce.configure(pce_opts)
        pce.set_indices(
            compute_hyperbolic_indices(var_trans.num_vars(), degree, 1.))

        nsamples = int(1e6)
        samples = lognorm.rvs(nsamples)[None, :]
        values = function(samples)

        ntrain_samples = 20
        train_samples = lognorm.rvs(ntrain_samples)[None, :]
        train_values = function(train_samples)
        # coef = solve_quantile_regression(
        #    0.5, np.log(train_samples), train_values, pce.basis_matrix,
        #   normalize_vals=True)
        coef = solve_least_squares_regression(
            np.log(train_samples), train_values, pce.basis_matrix,
            normalize_vals=True)
        pce.set_coefficients(coef)
        print(pce.mean(), values.mean())
        assert np.allclose(pce.mean(), values.mean(), rtol=1e-3)

    def test_get_univariate_quadrature_rules_from_variable(self):
        max_nsamples = 10
        nsamples = 5
        variable = IndependentMarginalsVariable(
            [stats.uniform(-1, 2), stats.norm(0, 1)])
        quad_rules = get_univariate_quadrature_rules_from_variable(
            variable, max_nsamples)

        x, w = quad_rules[0](nsamples)
        x_exact, w_exact = gauss_jacobi_pts_wts_1D(nsamples, 0, 0)
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)

        x, w = quad_rules[1](nsamples)
        x_exact, w_exact = gauss_hermite_pts_wts_1D(nsamples)
        assert np.allclose(x, x_exact)
        assert np.allclose(w, w_exact)

    def test_marginalize_function_1d(self):
        nsamples_1d = 10
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)]*2)
        quad_degrees = np.array([10])
        samples_ii = np.linspace(0, 1, nsamples_1d)
        from pyapprox.surrogates.polychaos.gpc import (
            _marginalize_function_1d)

        def fun(samples):
            return np.prod(samples**2, axis=0)[:, None]

        values = _marginalize_function_1d(
            fun, variable, quad_degrees, 0, samples_ii, qoi=0)
        assert np.allclose(values, samples_ii**2*1/3)

        def fun(samples):
            return np.cos(np.sum(samples, axis=0))[:, None]

        values = _marginalize_function_1d(
            fun, variable, quad_degrees, 0, samples_ii, qoi=0)
        assert np.allclose(values, np.sin(samples_ii+1)-np.sin(samples_ii))

    def test_marginalize_function_nd(self):
        from pyapprox.surrogates.polychaos.gpc import (
            _marginalize_function_nd)

        def fun(samples):
            vals = np.prod(samples**2, axis=0)[:, None]
            return vals

        nsamples_1d = 11
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)]*3)
        quad_degrees = np.array([3])
        samples = np.array([np.linspace(0, 1, nsamples_1d)]*2)
        indices = np.array([0, 1])
        values = _marginalize_function_nd(
            fun, variable, quad_degrees, indices, samples, qoi=0)
        assert np.allclose(values[:, 0], np.prod(samples**2, axis=0)*1/3)

        nsamples_1d = 11
        variable = IndependentMarginalsVariable([stats.uniform(0, 1)]*3)
        quad_degrees = np.array([3, 3])
        samples = np.array([np.linspace(0, 1, nsamples_1d)]*1)
        indices = np.array([1])
        values = _marginalize_function_nd(
            fun, variable, quad_degrees, indices, samples, qoi=0)
        assert np.allclose(values[:, 0], np.prod(samples**2, axis=0)*1/9)



if __name__ == "__main__":
    gpc_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGPC)
    unittest.TextTestRunner(verbosity=2).run(gpc_test_suite)
