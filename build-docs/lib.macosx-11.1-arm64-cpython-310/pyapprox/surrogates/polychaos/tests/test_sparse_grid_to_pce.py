
import unittest
import os
import pickle
import numpy as np
from functools import partial
from scipy import stats
import sympy as sp

from pyapprox.surrogates.interp.tests.test_sparse_grid import (
    MultilevelPolynomialModel, function_I)
from pyapprox.surrogates.orthopoly.quadrature import (
    leja_growth_rule,
    clenshaw_curtis_in_polynomial_order, clenshaw_curtis_rule_growth,
)
from pyapprox.surrogates.interp.sparse_grid import (
    get_1d_samples_weights, get_subspace_polynomial_indices,
    get_subspace_samples
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence
)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion,
    define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.interp.barycentric_interpolation import (
    compute_barycentric_weights_1d,
    multivariate_barycentric_lagrange_interpolation
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation,
    AffineTransform
)
from pyapprox.variables.density import beta_pdf_on_ab, gaussian_pdf
from pyapprox.util.utilities import hash_array
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    get_sparse_grid_univariate_leja_quadrature_rules_economical,
    variance_refinement_indicator, max_level_admissibility_function,
    CombinationSparseGrid
)
from pyapprox.surrogates.polychaos.sparse_grid_to_gpc import (
    convert_univariate_lagrange_basis_to_orthonormal_polynomials,
    convert_multivariate_lagrange_polys_to_orthonormal_polys,
    convert_sparse_grid_to_polynomial_chaos_expansion
)


class TestMultivariatePolynomials(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_convert_univariate_lagrange_basis_to_ortho_polynomials(self):
        level = 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]
        growth_rules = [clenshaw_curtis_rule_growth]
        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level])

        get_recursion_coefficients = partial(
            jacobi_recurrence, alpha=0., beta=0., probability=True)
        coeffs_1d = \
            convert_univariate_lagrange_basis_to_orthonormal_polynomials(
                samples_1d[0], get_recursion_coefficients)

        test_samples = np.linspace(-1, 1, 100)

        max_num_terms = samples_1d[0][-1].shape[0]
        recursion_coeffs = get_recursion_coefficients(max_num_terms)
        ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(
            test_samples, max_num_terms-1, recursion_coeffs)

        for ll in range(level+1):
            num_terms = coeffs_1d[ll].shape[0]
            barycentric_weights_1d = [
                compute_barycentric_weights_1d(samples_1d[0][ll])]
            values = np.eye((num_terms), dtype=float)
            lagrange_basis_vals =\
                multivariate_barycentric_lagrange_interpolation(
                    test_samples[np.newaxis,
                                 :], samples_1d[0][ll][np.newaxis, :],
                    barycentric_weights_1d, values, np.zeros(1, dtype=int))

            ortho_basis_vals = np.dot(
                ortho_basis_matrix[:, :num_terms], coeffs_1d[ll])
            # plt.plot(test_samples,ortho_basis_vals)
            # plt.show()
            assert np.allclose(ortho_basis_vals, lagrange_basis_vals)

    def test_convert_multivariate_lagrange_polys_to_orthonormal_polys(self):
        level, num_vars = 2, 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_vars
        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level]*num_vars)
        get_recursion_coefficients = [partial(
            jacobi_recurrence, alpha=0., beta=0., probability=True)]*num_vars
        coeffs_1d = [
            convert_univariate_lagrange_basis_to_orthonormal_polynomials(
                samples_1d[dd], get_recursion_coefficients[dd])
            for dd in range(num_vars)]

        def function(x): return (np.sum(x**2, axis=0) +
                                 np.prod(x, axis=0))[:, np.newaxis]

        subspace_indices = np.array([[1, 1], [0, 1], [2, 1]]).T
        for ii in range(subspace_indices.shape[1]):
            subspace_index = subspace_indices[:, ii]
            num_vars = subspace_index.shape[0]

            # subspace_samples_1d = \
            #    [samples_1d[subspace_index[ii]] for ii in range(num_vars)]
            # subspace_samples = cartesian_product(subspace_samples_1d)
            config_variables_idx = None
            poly_indices = get_subspace_polynomial_indices(
                subspace_index, growth_rules, config_variables_idx)
            subspace_samples = get_subspace_samples(
                subspace_index, poly_indices, samples_1d,
                config_variables_idx, unique_samples_only=False)

            subspace_values = function(subspace_samples)
            coeffs = convert_multivariate_lagrange_polys_to_orthonormal_polys(
                subspace_index, subspace_values, coeffs_1d, poly_indices,
                config_variables_idx)

            poly = PolynomialChaosExpansion()
            var_trans = define_iid_random_variable_transformation(
                stats.uniform(-1, 2), num_vars)
            poly_opts = define_poly_options_from_variable_transformation(
                var_trans)
            poly.configure(poly_opts)
            poly.set_indices(poly_indices)
            poly.set_coefficients(coeffs)

            # check the PCE is an interpolant
            poly_values = poly(subspace_samples)
            assert np.allclose(poly_values, subspace_values)

    def test_convert_multivariate_lagrange_polys_to_orthonormal_polys_mixed(
            self):
        level, num_vars = 2, 2
        alpha_stat, beta_stat = 5, 2
        variable = stats.beta(alpha_stat, beta_stat)
        beta_quad_rule = get_univariate_leja_quadrature_rule(
            variable, leja_growth_rule)
        quad_rules = [clenshaw_curtis_in_polynomial_order, beta_quad_rule]
        growth_rules = [clenshaw_curtis_rule_growth, leja_growth_rule]

        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level]*num_vars)
        get_recursion_coefficients = [partial(
            jacobi_recurrence, alpha=0., beta=0., probability=True), partial(
            jacobi_recurrence, alpha=beta_stat-1, beta=alpha_stat-1,
                probability=True)]
        coeffs_1d = [
            convert_univariate_lagrange_basis_to_orthonormal_polynomials(
                samples_1d[dd], get_recursion_coefficients[dd])
            for dd in range(num_vars)]

        def function(x): return (np.sum(x**2, axis=0) +
                                 np.prod(x, axis=0))[:, np.newaxis]

        poly = PolynomialChaosExpansion()
        univariate_variables = [
            stats.uniform(-1, 2), stats.beta(alpha_stat, beta_stat, -1, 2)]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)

        poly_opts = define_poly_options_from_variable_transformation(
                var_trans)
        poly.configure(poly_opts)

        subspace_indices = np.array([[1, 1], [0, 1], [2, 1]]).T
        for ii in range(subspace_indices.shape[1]):
            subspace_index = subspace_indices[:, ii]
            num_vars = subspace_index.shape[0]

            config_variables_idx = None
            poly_indices = get_subspace_polynomial_indices(
                subspace_index, growth_rules, config_variables_idx)
            subspace_samples = get_subspace_samples(
                subspace_index, poly_indices, samples_1d,
                config_variables_idx, unique_samples_only=False)

            subspace_values = function(subspace_samples)
            coeffs = convert_multivariate_lagrange_polys_to_orthonormal_polys(
                subspace_index, subspace_values, coeffs_1d, poly_indices,
                config_variables_idx)

            poly.set_indices(poly_indices)
            poly.set_coefficients(coeffs)

            # check the PCE is an interpolant
            poly_values = poly(subspace_samples)
            assert np.allclose(poly_values, subspace_values)

    def test_convert_sparse_grid_to_pce(self):
        num_vars = 2
        max_level = 2
        max_level_1d = [max_level]*(num_vars)
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)
        refinement_indicator = variance_refinement_indicator

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function_I)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)
        pce = convert_sparse_grid_to_polynomial_chaos_expansion(
            sparse_grid, pce_opts)

        # check that the sparse grid and the pce have the same polynomial terms
        assert len(sparse_grid.poly_indices_dict) == pce.indices.shape[1]
        for index in pce.indices.T:
            assert hash_array(index) in sparse_grid.poly_indices_dict

        pce_vals = pce(sparse_grid.samples)
        assert np.allclose(pce_vals, sparse_grid.values)

        filename = 'sparse-grid-test.pkl'
        sparse_grid.save(filename)
        with open(filename, 'rb') as f:
            sparse_grid_from_file = pickle.load(f)

        assert sparse_grid_from_file == sparse_grid
        os.remove(filename)

    def test_convert_sparse_grid_to_pce_mixed_basis(self):
        # self.help_convert_sparse_grid_to_pce_mixed_basis("pdf")
        self.help_convert_sparse_grid_to_pce_mixed_basis("christoffel")

    def help_convert_sparse_grid_to_pce_mixed_basis(self, leja_method):
        def function(x):
            return np.hstack((
                np.sum((x+1)**2, axis=0)[:, np.newaxis],
                np.sum((x-2)**2, axis=0)[:, np.newaxis]))

        num_vars = 2
        max_level = 5
        max_level_1d = [max_level]*(num_vars)

        alpha_stat, beta_stat = 2, 2
        univariate_variables = [
            stats.beta(alpha_stat, beta_stat, -1, 2), stats.norm()]
        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)

        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans, method=leja_method)

        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)
        refinement_indicator = variance_refinement_indicator

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function, growth_rules,
            unique_quadrule_indices=unique_quadrule_indices)
        sparse_grid.set_univariate_rules(quad_rules)
        sparse_grid.set_function(function)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        pce_opts = define_poly_options_from_variable_transformation(
            var_trans)
        pce = convert_sparse_grid_to_polynomial_chaos_expansion(
            sparse_grid, pce_opts)

        # check that the sparse grid and the pce have the same polynomial terms
        assert len(sparse_grid.poly_indices_dict) == pce.indices.shape[1]
        for index in pce.indices.T:
            assert hash_array(index) in sparse_grid.poly_indices_dict

        pce_vals = pce(sparse_grid.samples)
        assert np.allclose(pce_vals, sparse_grid.values)

        # num_validation_samples=int(1e6)
        # validation_samples = np.vstack((
        #   2*np.random.beta(alpha_stat,beta_stat,(1,num_validation_samples))-1,
        #     np.random.normal(0,1,(1,num_validation_samples))))
        # validation_values = function(validation_samples)
        # print (validation_values.mean(axis=0))

        x, y = sp.Symbol('x'), sp.Symbol('y')
        weight_function_x = beta_pdf_on_ab(alpha_stat, beta_stat, -1, 1, x)
        weight_function_y = gaussian_pdf(0, 1, y, package=sp)
        weight_function = weight_function_x*weight_function_y
        ranges = [-1, 1, -sp.oo, sp.oo]
        exact_mean = [
            float(sp.integrate(
                weight_function*((x+1)**2+(y+1)**2),
                (x, ranges[0], ranges[1]), (y, ranges[2], ranges[3]))),
            float(sp.integrate(
                weight_function*((x-2)**2+(y-2)**2),
                (x, ranges[0], ranges[1]), (y, ranges[2], ranges[3])))]

        assert np.allclose(exact_mean, pce.mean())

    def test_convert_multi_index_sparse_grid_to_pce(self):
        num_vars = 2
        num_levels = 3
        model = MultilevelPolynomialModel(num_levels)

        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars+1, num_validation_samples))
        validation_samples[-1, :] = num_levels-1
        validation_values = model(validation_samples)

        max_level = 5
        max_level_1d = [max_level]*(num_vars+1)
        max_level_1d[-1] = num_levels-1
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        def cost_function(x): return 1.
        refinement_indicator = variance_refinement_indicator

        sparse_grid = CombinationSparseGrid(num_vars+1)
        sparse_grid.set_function(model)
        sparse_grid.set_config_variable_index(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_interrogation_samples(validation_samples)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        # the pce will have no knowledge of configure variables.
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        pce_opts = define_poly_options_from_variable_transformation(var_trans)
        pce = convert_sparse_grid_to_polynomial_chaos_expansion(
            sparse_grid, pce_opts)

        # the sparse grid and the pce have the same poly_indices as indices
        # of the former include config variables
        # with configure variables sg and pce will not be an interpolant

        sg_values = sparse_grid(validation_samples)
        assert np.allclose(
            sg_values, sparse_grid.evaluate_at_interrogation_samples())
        pce_values = pce(validation_samples[:num_vars, :])
        assert np.allclose(pce_values, sg_values)
        assert np.allclose(pce_values, validation_values)


if __name__ == "__main__":
    multivariate_polynomials_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestMultivariatePolynomials)
    unittest.TextTestRunner(verbosity=2).run(
        multivariate_polynomials_test_suite)
    
