import unittest
from functools import partial
import numpy as np
from scipy import stats

from pyapprox.surrogates.polychaos.arbitrary_polynomial_chaos import (
    compute_moment_matrix_from_samples, APC, FPC,
    compute_moment_matrix_using_tensor_product_quadrature,
    compute_grammian_matrix_using_combination_sparse_grid,
    compute_coefficients_of_unrotated_basis,
    compute_polynomial_moments_using_tensor_product_quadrature,
    compute_rotation_from_moments_gram_schmidt,
    compute_rotation_from_moments_linear_system
)
from pyapprox.surrogates.polychaos.gpc import (
    define_poly_options_from_variable_transformation,
    PolynomialChaosExpansion
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.orthopoly.quadrature import gauss_jacobi_pts_wts_1D
from pyapprox.variables.density import tensor_product_pdf
from pyapprox.util.utilities import get_tensor_product_quadrature_rule
from pyapprox.surrogates.interp.mixture_model import (
    get_leja_univariate_quadrature_rules_of_beta_mixture,
    compute_grammian_of_mixture_models_using_sparse_grid_quadrature
)
from pyapprox.surrogates.orthopoly.quadrature import (
    leja_growth_rule, clenshaw_curtis_in_polynomial_order,
    clenshaw_curtis_rule_growth
)


class TestArbitraryPolynomialChaos(unittest.TestCase):

    def setUp(self):
        # np.set_printoptions(linewidth=200)
        # np.set_printoptions(precision=5)
        np.random.seed(1)

    def test_sample_based_apc_orthonormality(self):
        num_vars = 1
        alpha_stat = 2
        beta_stat = 5
        degree = 2

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(0, 1), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        random_var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat), num_vars)

        num_moment_samples = 10000
        moment_matrix_samples = generate_independent_random_samples(
            random_var_trans.variable, num_moment_samples)

        compute_moment_matrix_function = partial(
            compute_moment_matrix_from_samples, samples=moment_matrix_samples)

        pce = APC(compute_moment_matrix_function)
        pce.configure(pce_opts)

        num_samples = 10000
        samples = generate_independent_random_samples(
            random_var_trans.variable, num_samples)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)
        basis_matrix = pce.basis_matrix(samples)
        assert np.allclose(
            np.dot(basis_matrix.T, basis_matrix)/num_samples,
            np.eye(basis_matrix.shape[1]), atol=1e-1)

    def test_analytical_moment_based_apc_orthonormality_identity(self):
        """
        Test that when the correct orthonormal basis is used and integrated
        using quadrature that the rotation matrix is the identity. Test sets
        user domain to be different to canonical domain
        """
        num_vars = 1
        alpha_stat = 1
        beta_stat = 1
        degree = 2

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        random_var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat), num_vars)

        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            x = random_var_trans.map_from_canonical(x[np.newaxis, :])[
                0, :]
            return x, w

        # Test qr factorization to compute rotation matrix
        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=univariate_quadrature_rule)

        pce = APC(compute_moment_matrix_function)
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        assert np.allclose(pce.R_inv, np.eye(pce.R_inv.shape[0]))

        # Test cholesky factorization to compute rotation matrix
        def compute_grammian_function(basis_matrix_function, indices):
            num_samples = 10*degree
            basis_matrix = \
                compute_moment_matrix_using_tensor_product_quadrature(
                    basis_matrix_function, num_samples, num_vars,
                    univariate_quadrature_rule=univariate_quadrature_rule)
            return basis_matrix.T.dot(basis_matrix)

        pce_chol = APC(compute_grammian_function=compute_grammian_function)
        pce_chol.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce_chol.set_indices(indices)

        assert np.allclose(pce_chol.R_inv, np.eye(pce_chol.R_inv.shape[0]))
        # print (pce_chol.moment_matrix_cond,pce.moment_matrix_cond)

    def test_compute_moment_matrix_using_tensor_product_quadrature(self):
        """
        Test use of density_function in
        compute_moment_matrix_using_tensor_product_quadrature()
        """
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        degree = 3

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        random_var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat), num_vars)

        def univariate_pdf(x): return stats.beta.pdf(
                x, a=alpha_stat, b=beta_stat)

        density_function = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        def uniform_univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, 0, 0)
            x = (x+1.)/2.
            return x, w

        true_univariate_quadrature_rule = partial(
            gauss_jacobi_pts_wts_1D, alpha_poly=beta_stat-1,
            beta_poly=alpha_stat-1)

        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=uniform_univariate_quadrature_rule,
            density_function=density_function)

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, true_univariate_quadrature_rule,
            transform_samples=random_var_trans.map_from_canonical)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        apc = APC(compute_moment_matrix_function)
        apc.configure(pce_opts)
        apc.set_indices(indices)

        apc_basis_matrix = apc.basis_matrix(samples)

        # print(np.dot(apc_basis_matrix.T*weights,apc_basis_matrix))
        assert np.allclose(
            np.dot(apc_basis_matrix.T*weights, apc_basis_matrix),
            np.eye(apc_basis_matrix.shape[1]))

    def test_compute_moment_matrix_combination_sparse_grid(self):
        """
        Test use of density_function in
        compute_moment_matrix_using_tensor_product_quadrature()
        """
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        degree = 2

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        random_var_trans = define_iid_random_variable_transformation(
            stats.beta(alpha_stat, beta_stat), num_vars)

        def univariate_pdf(x):
            return stats.beta.pdf(x, a=alpha_stat, b=beta_stat)

        density_function = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        true_univariate_quadrature_rule = partial(
            gauss_jacobi_pts_wts_1D, alpha_poly=beta_stat-1,
            beta_poly=alpha_stat-1)

        quad_rule_opts = {'quad_rules': clenshaw_curtis_in_polynomial_order,
                          'growth_rules': clenshaw_curtis_rule_growth,
                          'unique_quadrule_indices': None}
        
        compute_grammian_function = partial(
            compute_grammian_matrix_using_combination_sparse_grid,
            var_trans=pce_var_trans, max_num_samples=100,
            density_function=density_function, quad_rule_opts=quad_rule_opts)

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, true_univariate_quadrature_rule,
            transform_samples=random_var_trans.map_from_canonical)

        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        pce.set_indices(indices)
        basis_matrix = pce.basis_matrix(samples)
        assert np.allclose(np.dot(basis_matrix.T*weights, basis_matrix),
                           compute_grammian_function(pce.basis_matrix, None))

        apc = APC(compute_grammian_function=compute_grammian_function)
        apc.configure(pce_opts)
        apc.set_indices(indices)

        apc_basis_matrix = apc.basis_matrix(samples)

        # print(np.dot(apc_basis_matrix.T*weights,apc_basis_matrix))
        assert np.allclose(
            np.dot(apc_basis_matrix.T*weights, apc_basis_matrix),
            np.eye(apc_basis_matrix.shape[1]))

    def test_compute_grammian_using_sparse_grid_quadrature(self):
        """
        Test compute_grammian_of_mixture_models_using_sparse_grid_quadrature()
        """
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        degree = 3

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        rv_params = [[alpha_stat, beta_stat]]
        mixtures, mixture_univariate_quadrature_rules = \
            get_leja_univariate_quadrature_rules_of_beta_mixture(
                rv_params, leja_growth_rule, None)

        compute_grammian_function = partial(
            compute_grammian_of_mixture_models_using_sparse_grid_quadrature,
            mixture_univariate_quadrature_rules=mixture_univariate_quadrature_rules,
            mixture_univariate_growth_rules=[leja_growth_rule],
            num_vars=num_vars)

        pce = APC(compute_grammian_function=compute_grammian_function)
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        # use Gauss quadrature for true distribution to integrate APC basis
        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            return x, w

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, univariate_quadrature_rule)

        basis_matrix = pce.basis_matrix(samples)
        # print (np.dot(basis_matrix.T*weights,basis_matrix))
        assert np.allclose(
            np.dot(basis_matrix.T*weights, basis_matrix),
            np.eye(basis_matrix.shape[1]))

    def test_get_unrotated_basis_coefficients(self):
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        degree = 3

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        def univariate_pdf(x):
            return stats.beta.pdf(x, a=alpha_stat, b=beta_stat)

        density_function = partial(
            tensor_product_pdf, univariate_pdfs=univariate_pdf)

        def uniform_univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, 0, 0)
            x = (x+1.)/2.
            return x, w

        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            x = (x+1.)/2.
            return x, w

        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=uniform_univariate_quadrature_rule,
            density_function=density_function)

        pce = APC(compute_moment_matrix_function)
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        # set pce coefficients randomly
        pce.coefficients = np.random.normal(0., 1., (indices.shape[1], 1))

        unrotated_basis_coefficients = compute_coefficients_of_unrotated_basis(
            pce.coefficients, pce.R_inv)

        num_samples = 10
        samples = np.random.uniform(0., 1., (num_vars, num_samples))
        true_values = pce(samples)
        values = np.dot(
            pce.unrotated_basis_matrix(samples), unrotated_basis_coefficients)
        assert np.allclose(values, true_values)

    def test_solve_linear_system_method(self):
        num_vars = 1
        alpha_stat = 2
        beta_stat = 2
        degree = 2

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            x = (x+1)/2.
            return x, w

        poly_moments = \
            compute_polynomial_moments_using_tensor_product_quadrature(
                pce.basis_matrix, 2*degree, num_vars,
                univariate_quadrature_rule)

        R_inv = compute_rotation_from_moments_linear_system(poly_moments)

        R_inv_gs = compute_rotation_from_moments_gram_schmidt(poly_moments)
        assert np.allclose(R_inv, R_inv_gs)

        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=univariate_quadrature_rule)

        apc = APC(compute_moment_matrix_function)
        apc.configure(pce_opts)
        apc.set_indices(indices)
        assert np.allclose(R_inv, apc.R_inv)

    def test_compute_rotation_using_moments(self):
        num_vars = 1
        alpha_stat = 2
        beta_stat = 2
        degree = 2

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)

        pce = PolynomialChaosExpansion()
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            x = (x+1)/2.
            return x, w

        poly_moments = \
            compute_polynomial_moments_using_tensor_product_quadrature(
                pce.basis_matrix, 2*degree, num_vars,
                univariate_quadrature_rule)

        apc1 = APC(moments=poly_moments)
        apc1.configure(pce_opts)
        apc1.set_indices(indices)

        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=univariate_quadrature_rule)

        apc2 = APC(compute_moment_matrix_function)
        apc2.configure(pce_opts)
        apc2.set_indices(indices)
        assert np.allclose(apc1.R_inv, apc2.R_inv)


class TestFramePolynomialChaos(unittest.TestCase):
    def setUp(self):
        pass

    def test_compute_moment_matrix_using_tensor_product_quadrature(self):
        """
        Test use of density_function in
        compute_moment_matrix_using_tensor_product_quadrature()
        """
        num_vars = 2
        alpha_stat = 2
        beta_stat = 5
        degree = 3

        pce_var_trans = define_iid_random_variable_transformation(
            stats.uniform(), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(
            pce_var_trans)
        pce_opts["truncation_tol"] = 1e-5

        def univariate_quadrature_rule(n):
            x, w = gauss_jacobi_pts_wts_1D(n, beta_stat-1, alpha_stat-1)
            x = (x+1)/2.
            return x, w

        compute_moment_matrix_function = partial(
            compute_moment_matrix_using_tensor_product_quadrature,
            num_samples=10*degree, num_vars=num_vars,
            univariate_quadrature_rule=univariate_quadrature_rule)

        pce = FPC(compute_moment_matrix_function)
        pce.configure(pce_opts)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        pce.set_indices(indices)

        samples, weights = get_tensor_product_quadrature_rule(
            degree+1, num_vars, univariate_quadrature_rule)

        basis_matrix = pce.basis_matrix(samples)
        # print np.dot(basis_matrix.T*weights,basis_matrix)
        assert np.allclose(
            np.dot(basis_matrix.T*weights, basis_matrix),
            np.eye(basis_matrix.shape[1]))


if __name__ == "__main__":
    apc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestArbitraryPolynomialChaos)
    unittest.TextTestRunner(verbosity=2).run(apc_test_suite)
    fpc_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFramePolynomialChaos)
    unittest.TextTestRunner(verbosity=2).run(fpc_test_suite)
