import unittest
from functools import partial
import numpy as np
import sympy as sp
from scipy import stats
import os

from pyapprox.surrogates.interp.sparse_grid import (
    update_1d_samples_weights_economical,
    get_1d_samples_weights, get_hierarchical_sample_indices,
    get_subspace_polynomial_indices, get_sparse_grid_samples_and_weights,
    get_subspace_samples, evaluate_sparse_grid, get_smolyak_coefficients,
    get_num_model_evaluations_from_samples, get_equivalent_cost,
    get_num_sparse_grid_samples, integrate_sparse_grid,
    tensor_product_lagrange_jacobian
)
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    CombinationSparseGrid,
    max_level_admissibility_function, mypriorityqueue,
    get_sparse_grid_univariate_leja_quadrature_rules_economical,
    variance_refinement_indicator, isotropic_refinement_indicator,
    update_smolyak_coefficients, surplus_refinement_indicator,
    insitu_update_sparse_grid_quadrature_rule,
    get_active_subspace_indices, extract_items_from_priority_queue,
    compute_hierarchical_surpluses_direct,
    extract_sparse_grid_quadrature_rule, compute_surpluses
)
from pyapprox.surrogates.interp.indexing import (
    set_difference, sort_indices_lexiographically,
    compute_hyperbolic_indices
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence, krawtchouk_recurrence
)
from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d
)
from pyapprox.surrogates.orthopoly.quadrature import (
    leja_growth_rule,
    clenshaw_curtis_in_polynomial_order, clenshaw_curtis_rule_growth,
    clenshaw_curtis_pts_wts_1D, gauss_quadrature
)
from pyapprox.surrogates.orthopoly.leja_quadrature import (
    get_univariate_leja_quadrature_rule,
    candidate_based_christoffel_leja_rule_1d
)
from pyapprox.util.utilities import (
    cartesian_product, hash_array, lists_of_arrays_equal, outer_product,
    allclose_unsorted_matrix_rows
)
from pyapprox.variables.density import beta_pdf_on_ab, gaussian_pdf
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
    )
from pyapprox.variables.transforms import (
    AffineBoundedVariableTransformation, AffineTransform
    )
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.interface.wrappers import WorkTrackingModel
from pyapprox.surrogates.interp.monomial import (
    evaluate_monomial, monomial_mean_uniform_variables,
    monomial_variance_uniform_variables, monomial_basis_matrix,
    evaluate_monomial_jacobian
)
from pyapprox.surrogates.interp.manipulate_polynomials import (
    get_indices_double_set)
from pyapprox.surrogates.interp.tensorprod import (
    canonical_univariate_piecewise_polynomial_quad_rule
)


class MultilevelPolynomialModel():
    def __init__(self, num_levels, return_work=False):
        self.num_levels = num_levels
        self.return_work = return_work
        self.ab = jacobi_recurrence(
            2*num_levels+1, alpha=0, beta=0, probability=True)
        self.coeff = 1./(10**np.arange(0, 2*num_levels+1))

    def __call__(self, samples):
        vals = []
        for ii in range(samples.shape[1]):
            level = samples[-1, ii]
            assert level.is_integer()
            level = int(level)
            assert level >= 0

            random_sample = samples[:-1, ii]
            basis_matrix = evaluate_orthonormal_polynomial_1d(
                np.asarray([random_sample.sum()]), level+1, self.ab)
            pp = np.dot(basis_matrix, self.coeff[:level+2])

            vals.append(pp)

        vals = np.asarray(vals)
        if self.return_work:
            vals = np.hstack(
                (vals, self.cost_function(samples[-1:, :])[:, np.newaxis]))
        return vals

    def cost_function(self, x):
        return x[0, :]+1.


class TestSparseGrid(unittest.TestCase):

    def test_update_1d_samples_weights_economical(self):
        num_vars = 3
        level = 2

        alpha_stat, beta_stat = 5, 2
        variable = stats.beta(alpha_stat, beta_stat)
        beta_quad_rule = get_univariate_leja_quadrature_rule(
            variable, leja_growth_rule)

        quad_rules_econ = [
            clenshaw_curtis_in_polynomial_order, beta_quad_rule]
        growth_rules_econ = [clenshaw_curtis_rule_growth, leja_growth_rule]
        unique_rule_indices = [[1], [0, 2]]

        levels = [level]*num_vars
        samples_1d_econ, weights_1d_econ = get_1d_samples_weights(
            quad_rules_econ, growth_rules_econ, levels, None,
            unique_rule_indices)

        quad_rules = [beta_quad_rule, clenshaw_curtis_in_polynomial_order,
                      beta_quad_rule]
        growth_rules = [leja_growth_rule, clenshaw_curtis_rule_growth,
                        leja_growth_rule]
        samples_1d, weights_1d = get_1d_samples_weights(
            quad_rules, growth_rules, levels)

        assert len(samples_1d_econ) == num_vars
        for ii in range(num_vars):
            assert len(samples_1d_econ[ii]) == len(samples_1d[ii])
            for jj in range(len(samples_1d[ii])):
                assert np.allclose(samples_1d[ii][jj], samples_1d_econ[ii][jj])
                assert np.allclose(weights_1d[ii][jj], weights_1d_econ[ii][jj])

        levels = [level+2]*num_vars
        samples_1d_econ, weights_1d_econ = \
            update_1d_samples_weights_economical(
                quad_rules_econ, growth_rules_econ,
                levels, samples_1d, weights_1d, None, unique_rule_indices)

        samples_1d, weights_1d = get_1d_samples_weights(
            quad_rules, growth_rules, levels, None)

        assert len(samples_1d_econ) == num_vars
        for ii in range(num_vars):
            assert len(samples_1d_econ[ii]) == len(samples_1d[ii])
            for jj in range(len(samples_1d[ii])):
                assert np.allclose(samples_1d[ii][jj], samples_1d_econ[ii][jj])
                assert np.allclose(weights_1d[ii][jj], weights_1d_econ[ii][jj])

        levels = [3, 5, 2]
        samples_1d_econ, weights_1d_econ = get_1d_samples_weights(
            quad_rules_econ, growth_rules_econ, levels, None,
            unique_rule_indices)

        quad_rules = [beta_quad_rule, clenshaw_curtis_in_polynomial_order,
                      beta_quad_rule]
        growth_rules = [leja_growth_rule, clenshaw_curtis_rule_growth,
                        leja_growth_rule]
        samples_1d, weights_1d = get_1d_samples_weights(
            quad_rules, growth_rules, levels)

        levels = np.asarray(levels)
        assert len(samples_1d_econ) == num_vars
        for dd in range(len(unique_rule_indices)):
            unique_rule_indices[dd] = np.asarray(
                unique_rule_indices[dd], dtype=int)
            max_level_dd = levels[unique_rule_indices[dd]].max()
            for ii in unique_rule_indices[dd]:
                assert len(samples_1d_econ[ii]) == max_level_dd+1

        for ii in range(num_vars):
            for jj in range(len(samples_1d[ii])):
                assert np.allclose(samples_1d[ii][jj], samples_1d_econ[ii][jj])
                assert np.allclose(weights_1d[ii][jj], weights_1d_econ[ii][jj])

    def test_get_hierarchical_sample_indices(self):
        num_vars = 4
        level = 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_vars
        samples_1d, __ = get_1d_samples_weights(quad_rules, growth_rules,
                                                [level]*num_vars)
        subspace_index = np.array([1, 0, 2, 0])
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules)
        config_variables_idx = None

        hier_indices = get_hierarchical_sample_indices(
            subspace_index, subspace_poly_indices,
            samples_1d, config_variables_idx)

        num_indices = 4
        indices = np.zeros((num_vars, num_indices), dtype=int)
        indices[0, 0] = 1
        indices[2, 0] = 3
        indices[0, 1] = 2
        indices[2, 1] = 3
        indices[0, 2] = 1
        indices[2, 2] = 4
        indices[0, 3] = 2
        indices[2, 3] = 4
        assert np.allclose(subspace_poly_indices[:, hier_indices], indices)

    def test_get_hierarchical_sample_indices_with_config_variables(self):
        num_config_vars = 1
        num_random_vars = 1
        num_vars = num_random_vars+num_config_vars
        level = 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_random_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_random_vars
        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level]*num_random_vars)
        subspace_index = np.array([0, 2])
        config_variables_idx = num_vars-num_config_vars
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules, config_variables_idx)

        hier_indices = get_hierarchical_sample_indices(
            subspace_index, subspace_poly_indices,
            samples_1d, config_variables_idx)

        indices = np.array([0])
        assert np.allclose(hier_indices, indices)

        subspace_index = np.array([1, 1])
        config_variables_idx = num_vars-num_config_vars
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules, config_variables_idx)

        hier_indices = get_hierarchical_sample_indices(
            subspace_index, subspace_poly_indices,
            samples_1d, config_variables_idx)

        indices = np.array([1, 2])
        assert np.allclose(hier_indices, indices)

        num_config_vars = 2
        num_random_vars = 2
        num_vars = num_random_vars+num_config_vars
        level = 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_random_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_random_vars
        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level]*num_random_vars)
        subspace_index = np.array([0, 0, 0, 2])
        config_variables_idx = num_vars-num_config_vars
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules, config_variables_idx)

        hier_indices = get_hierarchical_sample_indices(
            subspace_index, subspace_poly_indices,
            samples_1d, config_variables_idx)

        indices = np.zeros(1)
        # for some reason np.array([0])==np.array([]) in python so check length
        assert hier_indices.shape[0] == 1
        assert np.allclose(indices, hier_indices)

        subspace_index = np.array([1, 0, 0, 2])
        config_variables_idx = num_vars-num_config_vars
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules, config_variables_idx)

        hier_indices = get_hierarchical_sample_indices(
            subspace_index, subspace_poly_indices,
            samples_1d, config_variables_idx)

        indices = np.arange(1, 3)
        assert np.allclose(indices, hier_indices)

    def test_get_subspace_samples(self):
        num_vars = 4
        level = 2
        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_vars
        samples_1d, __ = get_1d_samples_weights(
            quad_rules, growth_rules, [level]*num_vars)
        subspace_index = np.array([1, 0, 2, 0])
        subspace_poly_indices = get_subspace_polynomial_indices(
            subspace_index, growth_rules)
        subspace_samples = get_subspace_samples(
            subspace_index, subspace_poly_indices, samples_1d)

        abscissa_1d = []
        for dd in range(num_vars):
            abscissa_1d.append(samples_1d[dd][subspace_index[dd]])
        samples = cartesian_product(abscissa_1d)
        assert np.allclose(subspace_samples, samples)

        subspace_index = np.array([1, 0, 2, 0])
        subspace_samples = get_subspace_samples(
            subspace_index, subspace_poly_indices, samples_1d,
            unique_samples_only=True)

        # there are two unique samples in each of the active variablces
        # so num_samples=4
        num_samples = 4
        samples = np.zeros((num_vars, num_samples))
        samples[0, 0] = samples_1d[0][1][1]
        samples[2, 0] = samples_1d[2][2][3]
        samples[0, 1] = samples_1d[0][1][2]
        samples[2, 1] = samples_1d[2][2][3]
        samples[0, 2] = samples_1d[0][1][1]
        samples[2, 2] = samples_1d[2][2][4]
        samples[0, 3] = samples_1d[0][1][2]
        samples[2, 3] = samples_1d[2][2][4]
        assert np.allclose(subspace_samples, samples)

    def test_sparse_grid_integration_clenshaw_curtis(self):
        num_vars = 4
        level = 3

        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth)

        poly_indices = data_structures[1]

        # plot_sparse_grid(samples,weights,poly_indices)
        # plt.show()

        J = np.arange(poly_indices.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0], 1))

        values = evaluate_monomial(poly_indices[:, J], coeffs, samples)
        assert np.allclose(np.dot(values[:, 0], weights),
                           monomial_mean_uniform_variables(
                               poly_indices[:, J], coeffs))

    def test_sparse_grid_integration_mixed_quadrature_rule(self):
        num_vars = 2
        level = 3

        alpha_stat, beta_stat = 5, 2
        variable = stats.beta(alpha_stat, beta_stat)
        beta_quad_rule = get_univariate_leja_quadrature_rule(
            variable, leja_growth_rule)

        quad_rules = [clenshaw_curtis_in_polynomial_order, beta_quad_rule]
        growth_rules = [clenshaw_curtis_rule_growth, leja_growth_rule]
        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, quad_rules, growth_rules)

        poly_indices = data_structures[1]

        # plot_sparse_grid(samples,weights,poly_indices)
        # plt.show()

        J = np.arange(poly_indices.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0]))

        x, y = sp.Symbol('x'), sp.Symbol('y')
        monomial_expansion = 0
        for ii in range(poly_indices.shape[1]):
            monomial_expansion +=\
                coeffs[ii]*x**poly_indices[0, ii]*y**poly_indices[1, ii]

        weight_function_x = 0.5
        weight_function_y = beta_pdf_on_ab(alpha_stat, beta_stat, -1, 1, y)
        weight_function = weight_function_x*weight_function_y
        ranges = [-1, 1, -1, 1]
        exact_mean = float(sp.integrate(
            monomial_expansion*weight_function,
            (x, ranges[0], ranges[1]), (y, ranges[2], ranges[3])))

        values = evaluate_monomial(poly_indices[:, J], coeffs, samples)
        sparse_grid_mean = np.dot(values[:, 0], weights)
        assert np.allclose(sparse_grid_mean, exact_mean)

    def test_sparse_grid_integration_arbitary_subspace_indices(self):
        num_vars = 3
        level = 4

        indices = compute_hyperbolic_indices(num_vars, level, 1.0)
        samples_1, weights_1, data_structures_1 =\
            get_sparse_grid_samples_and_weights(
                num_vars, level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth,
                sparse_grid_subspace_indices=indices)

        samples_2, weights_2, data_structures_2 =\
            get_sparse_grid_samples_and_weights(
                num_vars, level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth)

        poly_indices_1 = sort_indices_lexiographically(data_structures_1[1])
        poly_indices_2 = sort_indices_lexiographically(data_structures_2[1])
        assert np.allclose(poly_indices_1, poly_indices_2)
        assert np.allclose(np.sort(weights_1), np.sort(weights_2))
        assert allclose_unsorted_matrix_rows(samples_1.T, samples_2.T)

        J = np.arange(poly_indices_1.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0], 1))
        values = evaluate_monomial(poly_indices_1[:, J], coeffs, samples_1)
        assert np.allclose(np.dot(values[:, 0], weights_1),
                           monomial_mean_uniform_variables(
                               poly_indices_1[:, J], coeffs))

    def test_sparse_grid_integration_uniform_leja(self):
        num_vars = 2
        level = 11

        variable = stats.uniform()
        quadrature_rule = get_univariate_leja_quadrature_rule(
            variable, leja_growth_rule)

        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, quadrature_rule, leja_growth_rule)

        poly_indices = data_structures[1]

        J = np.arange(poly_indices.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0], 1))

        values = evaluate_monomial(poly_indices[:, J], coeffs, samples)
        assert np.allclose(np.dot(values[:, 0], weights),
                           monomial_mean_uniform_variables(
                               poly_indices[:, J], coeffs))

    def test_sparse_grid_integration_gaussian_leja(self):
        num_vars = 2
        level = 4

        variable = stats.norm()
        quadrature_rule = get_univariate_leja_quadrature_rule(
            variable, leja_growth_rule)

        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, quadrature_rule, leja_growth_rule)

        poly_indices = data_structures[1]

        J = np.arange(poly_indices.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0]))

        x, y = sp.Symbol('x'), sp.Symbol('y')
        monomial_expansion = 0
        for ii in range(poly_indices.shape[1]):
            monomial_expansion +=\
                coeffs[ii]*x**poly_indices[0, ii]*y**poly_indices[1, ii]

        def gaussian_pdf(mean, var, xx):
            return sp.exp(-(xx-mean)**2/(2*var)) / (2*sp.pi*var)**.5
        weight_function = gaussian_pdf(0, 1, x)*gaussian_pdf(0, 1, y)
        ranges = [-sp.oo, sp.oo, -sp.oo, sp.oo]
        exact_mean = float(sp.integrate(
            monomial_expansion*weight_function,
            (x, ranges[0], ranges[1]), (y, ranges[2], ranges[3])))

        values = evaluate_monomial(poly_indices[:, J], coeffs, samples)
        assert np.allclose(np.dot(values[:, 0], weights), exact_mean)

    def test_sparse_grid_integration_binomial_leja(self):
        num_vars = 2
        level = 5

        # precompute leja sequence
        num_trials, prob_success = [level+5, 0.5]
        assert num_trials >= leja_growth_rule(level)
        recursion_coeffs = krawtchouk_recurrence(
            num_trials, num_trials, prob_success)

        def generate_candidate_samples(num_samples):
            assert num_samples == num_trials+1
            return np.arange(0, num_trials+1)[np.newaxis, :]

        import tempfile
        temp_dirname = tempfile.mkdtemp()
        try:
            samples_filename = os.path.join(
                temp_dirname, 'binomial-leja-sequence-1d-ll-%d.npz' % (level))
            samples_filename = None
            quadrature_rule = partial(
                candidate_based_christoffel_leja_rule_1d, recursion_coeffs,
                generate_candidate_samples,
                num_trials+1,
                initial_points=np.atleast_2d(
                    [stats.binom.ppf(0.5, num_trials, prob_success)]),
                samples_filename=samples_filename,
                return_weights_for_all_levels=True)

            samples, weights, data_structures = \
                get_sparse_grid_samples_and_weights(
                    num_vars, level, quadrature_rule, leja_growth_rule)
        finally:
            import shutil
            shutil.rmtree(temp_dirname)

        poly_indices = data_structures[1]
        # plot_sparse_grid(samples,weights,poly_indices)
        # plt.show()

        J = np.arange(poly_indices.shape[1])
        coeffs = np.random.normal(0.0, 1.0, (J.shape[0]))
        values = evaluate_monomial(
            poly_indices[:, J], coeffs, samples/num_trials)

        validation_samples = cartesian_product(
            [np.arange(num_trials+1)]*num_vars)
        validation_values = evaluate_monomial(
            poly_indices[:, J], coeffs, validation_samples/num_trials)
        validation_weights = outer_product(
            [stats.binom.pmf(np.arange(num_trials+1),
                             num_trials, prob_success)]*num_vars)

        assert np.allclose(values[:, 0].dot(weights),
                           validation_values[:, 0].dot(validation_weights))

    def test_evaluate_sparse_grid_clenshaw_curtis(self):
        num_vars = 3
        level = 5

        quad_rules = [clenshaw_curtis_in_polynomial_order]*num_vars
        growth_rules = [clenshaw_curtis_rule_growth]*num_vars
        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, quad_rules, growth_rules)

        poly_indices_dict, poly_indices, subspace_indices,\
            smolyak_coefficients, subspace_poly_indices, samples_1d, \
            weights_1d, subspace_values_indices = data_structures

        J = np.arange(poly_indices.shape[1])
        monomial_indices = poly_indices[:, J]
        monomial_coeffs = np.random.normal(
            0.0, 1.0, (monomial_indices.shape[1], 1))

        values = evaluate_monomial(monomial_indices, monomial_coeffs, samples)

        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_validation_samples))

        validation_values = evaluate_monomial(
            monomial_indices, monomial_coeffs, validation_samples)

        # check sparse grid interpolates exactly sparse grid samples
        approx_values = evaluate_sparse_grid(
            samples, values, poly_indices_dict,
            subspace_indices, subspace_poly_indices, smolyak_coefficients,
            samples_1d, subspace_values_indices)
        assert np.allclose(approx_values, values)

        # test evaluation
        approx_values = evaluate_sparse_grid(
            validation_samples, values, poly_indices_dict,
            subspace_indices, subspace_poly_indices, smolyak_coefficients,
            samples_1d, subspace_values_indices)
        assert np.allclose(approx_values, validation_values)

        # test integration
        moments = integrate_sparse_grid(values, poly_indices_dict,
                                        subspace_indices,
                                        subspace_poly_indices,
                                        smolyak_coefficients, weights_1d,
                                        subspace_values_indices)

        assert np.allclose(
            moments[0, :], monomial_mean_uniform_variables(
                monomial_indices, monomial_coeffs))

        # test gradients
        approx_values, grads = evaluate_sparse_grid(
            validation_samples, values, poly_indices_dict,
            subspace_indices, subspace_poly_indices, smolyak_coefficients,
            samples_1d, subspace_values_indices, return_grad=True)

        assert np.allclose(approx_values, validation_values)

        validation_grads = evaluate_monomial_jacobian(
            monomial_indices, monomial_coeffs, validation_samples)
        for dd in range(num_vars):
            assert np.allclose(grads[:, :, dd], validation_grads[dd])

    def test_tensor_product_lagrange_jacobian(self):
        def fun(xx):
            return np.array([np.sum(xx**2, axis=0), np.sum(xx**1, axis=0)]).T

        def jac(xx):
            assert xx.shape[1] == 1
            jac = np.array([2*xx[:, 0], 1+xx[:, 0]*0])
            return jac

        levels = [1, 1]
        nvars = len(levels)
        # samples = np.random.uniform(-1, 1, (nvars, nsamples))
        samples = np.array([[0.5]*nvars]).T
        abscissa_1d = [clenshaw_curtis_pts_wts_1D(ll)[0] for ll in levels]
        grid_samples = cartesian_product(abscissa_1d)
        ident = np.eye(grid_samples.shape[1]) # return basis gradient
        basis_vals, basis_jacs = tensor_product_lagrange_jacobian(
            samples, abscissa_1d, ident)

        from pyapprox.util.utilities import approx_jacobian
        from pyapprox.surrogates.interp.barycentric_interpolation import (
            compute_barycentric_weights_1d,
            multivariate_barycentric_lagrange_interpolation)

        def basis_fun(xx):
            barycentric_weights_1d = [
                compute_barycentric_weights_1d(
                    abscissa_1d[dd], interval_length=2) for dd in range(nvars)]
            vals = multivariate_barycentric_lagrange_interpolation(
                xx, abscissa_1d, barycentric_weights_1d,
                np.eye(grid_samples.shape[1]), np.arange(nvars))
            return vals
        assert np.allclose(basis_fun(samples), basis_vals)
        jac_fd = approx_jacobian(lambda xx: basis_fun(xx).T, samples)
        assert np.allclose(jac_fd, basis_jacs)
        xx = np.random.uniform(0, 1, (nvars, 1))
        grid_values = fun(grid_samples)
        vals, jacs = tensor_product_lagrange_jacobian(
            xx, abscissa_1d, grid_values)
        assert np.allclose(vals, fun(xx))
        assert np.allclose(jacs[0], jac(xx))


def function_I(x):
    # define functions here so class sparse grid can be pickled
    return np.hstack((
        np.sum(np.exp(x), axis=0)[:, np.newaxis],
        10*np.sum(np.exp(x), axis=0)[:, np.newaxis]))


class TestAdaptiveSparseGrid(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_get_smolyak_coefficients(self):
        num_vars = 2
        level = 2
        samples, weights, data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth)

        subspace_indices = data_structures[2]
        smolyak_coeffs = get_smolyak_coefficients(subspace_indices)
        assert np.allclose(smolyak_coeffs, data_structures[3])

        num_vars = 3
        level = 5
        __, __, data_structures = get_sparse_grid_samples_and_weights(
            num_vars, level, clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth)

        subspace_indices = data_structures[2]
        smolyak_coeffs = get_smolyak_coefficients(subspace_indices)
        assert np.allclose(smolyak_coeffs, data_structures[3])

        num_vars = 2
        level = 2
        __, __, data_structures = get_sparse_grid_samples_and_weights(
            num_vars, level, clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth)

        new_index = np.array([level+1, 0]).T
        subspace_indices = np.hstack(
            (data_structures[2], new_index[:, np.newaxis]))
        smolyak_coeffs = get_smolyak_coefficients(subspace_indices)

        subspace_indices_dict = dict()
        for ii in range(subspace_indices.shape[1]):
            subspace_indices_dict[tuple(list(subspace_indices[:, ii]))] =\
                smolyak_coeffs[ii]

        assert np.allclose(subspace_indices_dict[tuple(list(new_index))], 1.0)
        assert np.allclose(subspace_indices_dict[(level, 0)], 0.0)

    def test_update_smolyak_coefficients(self):

        num_vars = 2
        level = 5

        __, __, data_structures = get_sparse_grid_samples_and_weights(
            num_vars, level, clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth)

        subspace_indices = data_structures[2]
        smolyak_coeffs = data_structures[3]

        new_index = np.array([level+1, 0]).T

        subspace_indices = np.hstack(
            (subspace_indices, new_index[:, np.newaxis]))
        smolyak_coeffs = np.append(smolyak_coeffs, 0.)

        smolyak_coeffs = update_smolyak_coefficients(
            new_index, subspace_indices, smolyak_coeffs)

        subspace_indices_dict = dict()
        for ii in range(subspace_indices.shape[1]):
            subspace_indices_dict[tuple(list(subspace_indices[:, ii]))] =\
                smolyak_coeffs[ii]

        assert np.allclose(subspace_indices_dict[tuple(list(new_index))], 1.0)
        assert np.allclose(subspace_indices_dict[(level, 0)], 0.0)

    def test_update_smolyak_coefficients_iteratively(self):
        """
        Test that when we update an isotropic sparse grid iteratively starting
        at one level we get the coefficients of the isotropic grid at the
        next level
        """
        num_vars = 3
        level = 2

        __, __, data_structures_l = get_sparse_grid_samples_and_weights(
            num_vars, level, clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth)

        subspace_indices_l = data_structures_l[2]
        smolyak_coeffs_l = data_structures_l[3]

        next_level = level+1
        __, __, data_structures_lp1 = get_sparse_grid_samples_and_weights(
            num_vars, next_level, clenshaw_curtis_in_polynomial_order,
            clenshaw_curtis_rule_growth)

        subspace_indices_lp1 = data_structures_lp1[2]
        smolyak_coeffs_lp1 = data_structures_lp1[3]

        # get indices in lp1 but not in l
        new_indices = set_difference(subspace_indices_l, subspace_indices_lp1)

        # udpate lowest level sparse grid iteratively
        subspace_indices = subspace_indices_l.copy()
        smolyak_coeffs = smolyak_coeffs_l.copy()

        for ii in range(new_indices.shape[1]):
            subspace_indices = np.hstack(
                (subspace_indices, new_indices[:, ii:ii+1]))
            smolyak_coeffs = np.append(smolyak_coeffs, 0.)
            smolyak_coeffs = update_smolyak_coefficients(
                new_indices[:, ii], subspace_indices, smolyak_coeffs)

        # from pyapprox.util.visualization import plot_2d_indices
        # import matplotlib.pyplot as plt
        # plot_2d_indices(subspace_indices,smolyak_coeffs)
        # plt.figure()
        # plot_2d_indices(subspace_indices_lp1,smolyak_coeffs_lp1)
        # plt.show()

        # Sparse grid data structures for isotropic sparse grids do not
        # store coefficients of subspace with smolyak coefficients that are
        # zero.
        # The order of iteratively built index set may be different to that
        # of isotropic sparse grid so sort indices and compare ignoring
        # subspace that have smolyak coefficients that are zero.
        II = np.where(smolyak_coeffs > 0)[0]
        JJ = np.where(smolyak_coeffs_lp1 > 0)[0]
        assert smolyak_coeffs[II].shape[0] == smolyak_coeffs_lp1[JJ].shape[0]

        assert set_difference(
            subspace_indices_lp1[:, JJ], subspace_indices[:, II]).shape[1] == 0

        assert set_difference(
            smolyak_coeffs_lp1[JJ], smolyak_coeffs[II]).shape[0] == 0

    def test_hierarchical_surplus_equivalence(self):
        num_vars = 2
        max_level = 4

        refinement_indicator = isotropic_refinement_indicator

        max_level_1d = [max_level]*(num_vars)
        # max_level_1d=[max_level]*(num_vars+num_config_vars)
        # max_level_1d[-1]=num_model_levels-1
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function_I)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            items, sparse_grid.active_subspace_queue = \
                extract_items_from_priority_queue(
                    sparse_grid.active_subspace_queue)
            for item in items:
                priority, error, ii = item
                active_subspace_index =\
                    sparse_grid.subspace_indices[:, ii]

                surpluses1 = compute_hierarchical_surpluses_direct(
                    active_subspace_index, sparse_grid)

                surpluses2, hier_indices = compute_surpluses(
                    active_subspace_index, sparse_grid, hierarchical=True)

                assert np.allclose(surpluses1, surpluses2)

            sparse_grid.refine()

    def test_variable_transformation(self):
        num_vars = 2
        max_level = 4
        ranges = [-1, 1, -1, 1]
        ranges = [0, 1, 0, 1]
        w = np.prod([1./(ranges[2*ii+1]-ranges[2*ii])
                     for ii in range(num_vars)])

        x, y = sp.Symbol('x'), sp.Symbol('y')
        exact_mean = float(sp.integrate(
            (x**3+x*y+y**2)*w, (x, ranges[0], ranges[1]),
            (y, ranges[2], ranges[3])))
        exact_variance = float(sp.integrate(
            (x**3+x*y+y**2)**2*w, (x, ranges[0], ranges[1]),
            (y, ranges[2], ranges[3]))-exact_mean**2)

        def function(x): return np.array(
            [x[0, :]**3+x[0, :]*x[1, :]+x[1, :]**2]).T

        canonical_ranges = [(-1)**(ii+1) for ii in range(2*num_vars)]
        var_trans = AffineBoundedVariableTransformation(
            canonical_ranges, ranges)

        refinement_indicator = isotropic_refinement_indicator

        max_level_1d = [max_level]*num_vars
        max_num_sparse_grid_samples = 100
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function, var_trans)
        sparse_grid.build()

        moments = sparse_grid.moments()[:, 0]
        exact_moments = np.asarray([exact_mean, exact_variance])
        # print(moments,exact_moments)
        assert np.allclose(moments, exact_moments)

        num_validation_samples = 10
        validation_samples = np.random.uniform(
            -1, 1, (num_vars, num_validation_samples))
        validation_samples = var_trans.map_from_canonical(
            validation_samples)

        values = sparse_grid(validation_samples)
        validation_values = function(validation_samples)
        assert np.allclose(values, validation_values)

        values = sparse_grid.evaluate_using_all_data(validation_samples)
        assert np.allclose(values, validation_values)

    def test_hierarchical_refinement_indicator(self):
        num_vars = 2
        max_level = 4

        def function(x): return np.array(
            [np.sum(1+np.exp(x), axis=0), 1+np.sum(x**12, axis=0)]).T

        refinement_indicator = partial(
            surplus_refinement_indicator, hierarchical=True, norm_order=1)

        max_level_1d = [max_level]*num_vars
        max_num_sparse_grid_samples = 100
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            items, sparse_grid.active_subspace_queue = \
                extract_items_from_priority_queue(
                    sparse_grid.active_subspace_queue)
            for item in items:
                priority, error, ii = item
                active_subspace_index =\
                    sparse_grid.subspace_indices[:, ii]
                if np.count_nonzero(active_subspace_index) > 1:
                    # subspaces with interactions terms will be
                    # added because we have set no termination condition
                    # based upon total error, however we can check
                    # the hierarchical surpluses of these subspaces is zero
                    # error = 0. <==> priority=inf
                    # assert priority==np.inf
                    pass
                elif active_subspace_index.max() == 2:
                    new_samples = sparse_grid.samples_1d[0][2][3:5]
                    new_weights = sparse_grid.weights_1d[0][2][3:5]
                    nsamples = new_samples.shape[0]
                    # error = np.linalg.norm(
                    #     new_samples**2-new_samples**8)/np.sqrt(nsamples)
                    # error = max(error,np.linalg.norm(
                    #     new_samples**2-new_samples**12)/np.sqrt(nsamples))
                    error = np.abs(
                        np.sum((new_samples**2-new_samples**8)*new_weights))
                    error = max(np.abs(
                        np.sum((new_samples**2-new_samples**12)*new_weights)),
                        error)
                    cost = nsamples*1.
                    assert np.allclose(priority, -error/cost)
                # print active_subspace_index,priority
            sparse_grid.refine()

        # plot_adaptive_sparse_grid_2d(sparse_grid,plot_grid=True)
        # plt.show()

    def test_variance_refinement_indicator(self):
        num_vars = 2
        max_level = 4

        def function(x): return np.array(
            [1+np.sum(np.exp(x), axis=0), 1+np.sum(x**12, axis=0)]).T

        # refinement_indicator=isotropic_refinement_indicator
        refinement_indicator = variance_refinement_indicator

        max_level_1d = [max_level]*num_vars
        max_num_sparse_grid_samples = 100
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function)

        step = 0
        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            items, sparse_grid.active_subspace_queue = \
                extract_items_from_priority_queue(
                    sparse_grid.active_subspace_queue)
            for item in items:
                priority, error, ii = item
                active_subspace_index =\
                    sparse_grid.subspace_indices[:, ii]
                if np.count_nonzero(active_subspace_index) > 1:
                    # subspaces with interactions terms will be
                    # added because we have set no termination condition
                    # based upon total error, however we can check
                    # the hierarchical surpluses of these subspaces is zero
                    # error = 0. <==> priority=inf
                    # assert priority==np.inf
                    pass
                # print active_subspace_index,priority
            sparse_grid.refine()

            # plot_adaptive_sparse_grid_2d(sparse_grid,plot_grid=True)
            # plt.savefig('adaptive-refinement-plot-step-%d'%step)
            # plt.show()
            step += 1

    def test_adaptive_combination_technique(self):
        num_vars = 2
        max_level = 5

        __, __, isotropic_data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, max_level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth)

        poly_indices = isotropic_data_structures[1]
        # monomial_idx = np.arange(poly_indices.shape[1])
        # for variance computation to be exact form a polynomial whose
        # indices form the half set of the sparse grid polynomial indices
        monomial_idx = []
        for ii in range(poly_indices.shape[1]):
            if poly_indices[:, ii].sum() < max_level:
                monomial_idx.append(ii)
        monomial_idx = np.asarray(monomial_idx)
        monomial_indices = poly_indices[:, monomial_idx]
        monomial_coeffs = np.random.normal(
            0.0, 1.0, (monomial_idx.shape[0], 1))

        def function(x): return evaluate_monomial(
            monomial_indices, monomial_coeffs, x)
        # function = lambda x: np.sum(x**8,axis=0)[:,np.newaxis]

        num_validation_samples = 1000
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_validation_samples))

        validation_values = function(validation_samples)

        max_level_1d = None
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)
        refinement_indicator = isotropic_refinement_indicator
        refinement_indicator = variance_refinement_indicator

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function)
        sparse_grid.build()

        assert (
            len(isotropic_data_structures[0]) == len(
                sparse_grid.poly_indices_dict))
        # assert isotropic_data_structures[0]==data_structures[0] will not work
        # keys will be the same but not idx
        for key in isotropic_data_structures[0]:
            assert key in sparse_grid.poly_indices_dict

        II = np.where(sparse_grid.smolyak_coefficients > 0)[0]
        JJ = np.where(isotropic_data_structures[3] > 0)[0]
        assert (isotropic_data_structures[2][:, JJ].shape ==
                sparse_grid.subspace_indices[:, II].shape)
        assert set_difference(
            isotropic_data_structures[2][:, JJ],
            sparse_grid.subspace_indices[:, II]).shape[1] == 0

        # check sparse grid interpolates exactly sparse grid samples
        approx_values = sparse_grid(sparse_grid.samples)
        assert np.allclose(approx_values, sparse_grid.values)

        approx_values = sparse_grid(validation_samples)
        assert np.allclose(approx_values, validation_values)

        moments = sparse_grid.moments()
        assert np.allclose(
            moments[0, :], monomial_mean_uniform_variables(
                monomial_indices, monomial_coeffs))

        assert np.allclose(
            moments[1, :], monomial_variance_uniform_variables(
                monomial_indices, monomial_coeffs))

        num_samples = get_num_sparse_grid_samples(
            sparse_grid.subspace_poly_indices_list,
            sparse_grid.smolyak_coefficients)
        assert np.allclose(num_samples, sparse_grid.values.shape[0])

    def test_evaluate_using_all_data(self):
        """
        Check that for a level 0 grid with all level 1 subspaces active
        that these active subspaces can be included when evaluating
        the grid without affecting refinement
        """
        num_vars = 2
        max_level = 1

        sparse_grid_samples, __, isotropic_data_structures = \
            get_sparse_grid_samples_and_weights(
                num_vars, max_level, clenshaw_curtis_in_polynomial_order,
                clenshaw_curtis_rule_growth)

        poly_indices = isotropic_data_structures[1]
        # monomial_idx = np.arange(poly_indices.shape[1])
        # for variance computation to be exact form a polynomial whose
        # indices form the half set of the sparse grid polynomial indices
        monomial_idx = []
        for ii in range(poly_indices.shape[1]):
            # if poly_indices[:,ii].sum()<max_level:
            monomial_idx.append(ii)
        monomial_idx = np.asarray(monomial_idx)
        monomial_indices = poly_indices[:, monomial_idx]
        monomial_coeffs = np.random.normal(0.0, 1.0, (monomial_idx.shape[0]))
        def function(x): return evaluate_monomial(
            monomial_indices, monomial_coeffs, x)

        num_validation_samples = 1000
        validation_samples = np.random.uniform(
            -1., 1., (num_vars, num_validation_samples))

        poly_indices_dict, poly_indices, subspace_indices,\
            smolyak_coefficients, subspace_poly_indices, samples_1d, \
            weights_1d, subspace_values_indices = isotropic_data_structures

        sparse_grid_values = function(sparse_grid_samples)

        sparse_grid_validation_values = evaluate_sparse_grid(
            validation_samples, sparse_grid_values,
            poly_indices_dict, subspace_indices, subspace_poly_indices,
            smolyak_coefficients, samples_1d,
            subspace_values_indices)

        max_level_1d = None
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)
        refinement_indicator = isotropic_refinement_indicator

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

            coef = sparse_grid.smolyak_coefficients.copy()
            pairs, sparse_grid.active_subspace_queue = \
                extract_items_from_priority_queue(
                    sparse_grid.active_subspace_queue)
            approx_values = sparse_grid.evaluate_using_all_data(
                validation_samples)
            # check above function does not change smolyak coefficients
            assert np.allclose(coef, sparse_grid.smolyak_coefficients)
            # check evaluate_using_all_data does not change priority queue
            pairs_new, sparse_grid.active_subspace_queue = \
                extract_items_from_priority_queue(
                    sparse_grid.active_subspace_queue)
            assert pairs_new == pairs
            # check sparse grid values are the same as those obtained using
            # level 1 isotropic sparse grid
            assert np.allclose(approx_values, sparse_grid_validation_values)

        num_samples = get_num_sparse_grid_samples(
            sparse_grid.subspace_poly_indices_list,
            sparse_grid.smolyak_coefficients)
        assert np.allclose(num_samples, sparse_grid.values.shape[0])

    def test_extract_items_from_priority_queue(self):
        pairs = [(0., 0), (10., 1), (2, 2)]
        pqueue = mypriorityqueue()
        for ii in range(len(pairs)):
            pqueue.put(pairs[ii])
        pairs_new, pqueue_new = extract_items_from_priority_queue(pqueue)

        sorted_idx = sorted(np.arange(len(pairs)), key=lambda x: pairs[x])
        for ii in range(len(pairs)):
            assert pairs_new[ii] == pairs[sorted_idx[ii]]

    def test_nested_refinement(self):
        """
        """
        num_vars = 2
        max_level = 10

        # function = lambda x: (
        #    np.sum(np.exp(x),axis=0)+x[0,:]**3*x[1,:]**3)[:,np.newaxis]
        def function(x): return np.hstack((
            np.sum(np.exp(x), axis=0)[:, np.newaxis],
            10*np.sum(np.exp(x), axis=0)[:, np.newaxis]))

        max_level_1d = None
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

        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)
        sparse_grid.set_function(function)

        num_refinement_steps = 10
        priority_dict = dict()
        active_subspace_indices, II = get_active_subspace_indices(
            sparse_grid.active_subspace_indices_dict,
            sparse_grid.subspace_indices)
        for ii in range(active_subspace_indices.shape[1]):
            subspace_index = active_subspace_indices[:, ii]
            # use dummy value of 1 for refinement indicator
            priority, error = refinement_indicator(
                subspace_index, 1, sparse_grid)
            key = hash_array(subspace_index)
            if key in priority_dict:
                assert np.allclose(priority_dict[key], priority)
            else:
                priority_dict[key] = priority

        for jj in range(num_refinement_steps):
            sparse_grid.refine()
            active_subspace_indices, II = get_active_subspace_indices(
                sparse_grid.active_subspace_indices_dict,
                sparse_grid.subspace_indices)
            for ii in range(active_subspace_indices.shape[1]):
                subspace_index = active_subspace_indices[:, ii]
                # use dummy value of 1 for refinement indicator
                priority, error = refinement_indicator(
                    subspace_index, 1, sparse_grid)
                key = hash_array(subspace_index)
                if key in priority_dict:
                    assert np.allclose(priority_dict[key], priority)
                else:
                    priority_dict[key] = priority

    def test_polynomial_quadrature_order_accuracy(self):
        level = 2
        alpha = 0
        beta = 0
        cc_x, cc_w = clenshaw_curtis_pts_wts_1D(level)

        degree = 9
        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        def function(x):
            p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
            # evaluate polynomial with all coefficients equal to one
            return p.sum(axis=1)

        gauss_x, gauss_w = gauss_quadrature(ab, degree+1)

        # compute interpolant using Clenshaw-Curtis samples
        vandermonde = evaluate_orthonormal_polynomial_1d(
            cc_x, cc_x.shape[0]-1, ab)
        values = function(cc_x)
        coeff = np.linalg.lstsq(vandermonde, values, rcond=None)[0]

        # integrate interpolant using Gauss-Jacobi quadrature
        vandermonde = evaluate_orthonormal_polynomial_1d(
            gauss_x, cc_x.shape[0]-1, ab)
        interp_values = np.dot(vandermonde, coeff)

        gauss_mean = np.dot(interp_values, gauss_w)
        gauss_variance = np.dot(interp_values**2, gauss_w)-gauss_mean**2

        cc_mean = np.dot(values, cc_w)

        pce_mean = coeff[0]
        pce_variance = np.sum(coeff[1:]**2)

        assert np.allclose(gauss_mean, cc_mean)
        assert np.allclose(gauss_mean, pce_mean)
        assert np.allclose(gauss_variance, pce_variance)

    def economical_quad_rules_helper(
            self, selected_variables_idx,
            all_univariate_variables, all_sp_variables,
            all_ranges, all_weight_functions,
            max_level, growth_rules=None):

        def function(x):
            vals = np.hstack((
                np.sum((x+1)**2, axis=0)[:, np.newaxis],
                np.sum((x-2)**2, axis=0)[:, np.newaxis]))
            return vals

        univariate_variables = []
        variables = []
        ranges = np.empty(2*selected_variables_idx.shape[0])
        weight_functions = []
        for ii in range(len(selected_variables_idx)):
            index = selected_variables_idx[ii]
            univariate_variables.append(all_univariate_variables[index])
            variables.append(all_sp_variables[index])
            ranges[2*ii:2*(ii+1)] = all_ranges[2*index:2*(index+1)]
            weight_functions.append(all_weight_functions[index])

        variable = IndependentMarginalsVariable(univariate_variables)
        var_trans = AffineTransform(variable)

        num_vars = len(univariate_variables)
        max_level_1d = [max_level]*(num_vars)

        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans, growth_rules)

        assert len(quad_rules) == len(growth_rules)

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
        sparse_grid.set_function(function, var_trans)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        # plt.plot(sparse_grid.samples[0,:],sparse_grid.samples[1,:],'o')
        # plt.show()
        # multivariate integration takes to long so break up into 1d integrals
        # weight_function =  weight_function_w*weight_function_x*\
            #    weight_function_y*weight_function_z
        exact_mean = np.zeros(2)
        for ii in range(len(variables)):
            exact_mean[0] += float(
                sp.integrate(weight_functions[ii]*(variables[ii]+1)**2,
                             (variables[ii], ranges[2*ii], ranges[2*ii+1])))
            assert np.allclose(
                1.,
                float(sp.integrate(
                    weight_functions[ii],
                    (variables[ii], ranges[2*ii], ranges[2*ii+1]))))
            exact_mean[1] += float(
                sp.integrate(weight_functions[ii]*(variables[ii]-2)**2,
                             (variables[ii], ranges[2*ii], ranges[2*ii+1])))

        assert np.allclose(exact_mean, sparse_grid.moments()[0])
        return unique_quadrule_indices

    def test_economical_quad_rules(self):

        alpha_stat1, beta_stat1 = 2, 2
        alpha_stat2, beta_stat2 = 3, 3
        beta_var0 = {'var_type': 'beta', 'range': [0, 1],
                     'alpha_stat': alpha_stat1, 'beta_stat': beta_stat1}
        beta_var1 = {'var_type': 'beta', 'range': [-1, 1],
                     'alpha_stat': alpha_stat1, 'beta_stat': beta_stat1}
        beta_var2 = {'var_type': 'beta', 'range': [-1, 1],
                     'alpha_stat': alpha_stat2, 'beta_stat': beta_stat2}
        gaussian_var = {'var_type': 'gaussian', 'mean': 0., 'variance': 1.}
        univariate_variables = [beta_var0, beta_var1, beta_var2,
                                gaussian_var, beta_var1]
        univariate_variables = [
            stats.beta(alpha_stat1, beta_stat1), stats.beta(
                alpha_stat1, beta_stat1, -1, 2),
            stats.beta(alpha_stat2, beta_stat2, -1, 2), stats.norm(),
            stats.beta(alpha_stat1, beta_stat1, -1, 2)]

        v, w, x, y = sp.Symbol('v'), sp.Symbol(
            'w'), sp.Symbol('x'), sp.Symbol('y')
        z = sp.Symbol('z')
        weight_function_v = beta_pdf_on_ab(alpha_stat1, beta_stat1, 0, 1, v)
        weight_function_w = beta_pdf_on_ab(alpha_stat1, beta_stat1, -1, 1, w)
        weight_function_x = beta_pdf_on_ab(alpha_stat2, beta_stat2, -1, 1, x)
        weight_function_y = gaussian_pdf(0, 1, y, package=sp)
        weight_function_z = beta_pdf_on_ab(alpha_stat1, beta_stat1, -1, 1, z)
        ranges = [0, 1, -1, 1, -1, 1, -sp.oo, sp.oo, -1, 1]
        sp_variables = [v, w, x, y, z]
        weight_functions = [
            weight_function_v, weight_function_w, weight_function_x,
            weight_function_y, weight_function_z]

        selected_variables_idx = np.asarray([0, 1])
        unique_quadrule_indices = self.economical_quad_rules_helper(
            selected_variables_idx, univariate_variables, sp_variables,
            ranges, weight_functions, 1, clenshaw_curtis_rule_growth)
        assert lists_of_arrays_equal(unique_quadrule_indices, [[0, 1]])

        # assumes that only one type of quadrule can be specified
        selected_variables_idx = np.asarray([0, 1])
        self.assertRaises(
            Exception, self.economical_quad_rules_helper,
            selected_variables_idx, univariate_variables, sp_variables,
            ranges, weight_functions, 2,
            [clenshaw_curtis_rule_growth, leja_growth_rule])

        selected_variables_idx = np.asarray([2, 3, 0, 1])
        unique_quadrule_indices = self.economical_quad_rules_helper(
            selected_variables_idx, univariate_variables, sp_variables,
            ranges, weight_functions, 2)
        assert lists_of_arrays_equal(
            unique_quadrule_indices, [[0], [1], [2, 3]])

        selected_variables_idx = np.asarray([1, 2, 3, 4])
        unique_quadrule_indices = self.economical_quad_rules_helper(
            selected_variables_idx, univariate_variables, sp_variables,
            ranges, weight_functions, 2)
        assert lists_of_arrays_equal(
            unique_quadrule_indices, [[0, 3], [1], [2]])

        selected_variables_idx = np.asarray([0, 1, 2, 3, 4])
        unique_quadrule_indices = self.economical_quad_rules_helper(
            selected_variables_idx, univariate_variables, sp_variables,
            ranges, weight_functions, 2)
        assert lists_of_arrays_equal(
            unique_quadrule_indices, [[0, 1, 4], [2], [3]])

    def test_error_based_stopping_criteria(self):
        num_vars = 2
        level = 2
        indices = compute_hyperbolic_indices(num_vars, level, .5)

        variable = stats.uniform(-1, 2)
        var_trans = define_iid_random_variable_transformation(
            variable, num_vars)

        # to generate quadrature rule that integrates all inner products
        # must integrate the double set
        double_set_indices = get_indices_double_set(indices)
        basis_matrix_function = partial(
            monomial_basis_matrix, double_set_indices)

        growth_rule = leja_growth_rule
        univariate_quadrature_rule = get_univariate_leja_quadrature_rule(
            variable, growth_rule)
        max_level_1d = None
        max_num_sparse_grid_samples = None
        admissibility_function = partial(
            max_level_admissibility_function, np.inf, max_level_1d,
            max_num_sparse_grid_samples, 1e-12)

        # variance based refinement indicator will not correctly exit with
        # leja_growth_rule because integral of an odd basis is zero and so
        # will not contribute to error
        # refinement_indicator = partial(
        #     variance_refinement_indicator, normalize=False, mean_only=True)
        refinement_indicator = partial(
            surplus_refinement_indicator, normalize=False)

        sparse_grid = CombinationSparseGrid(num_vars)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function, growth_rule)
        sparse_grid.set_univariate_rules(univariate_quadrature_rule)
        sparse_grid.set_function(basis_matrix_function)
        sparse_grid.build()
        # for ii in range(sparse_grid.subspace_indices.shape[1]):
        #     print(ii)
        #     print(sparse_grid.subspace_indices[:, ii])
        #     print(sparse_grid.subspace_poly_indices_list[ii])

        samples, weights = extract_sparse_grid_quadrature_rule(sparse_grid)
        samples = var_trans.map_from_canonical(samples)

        basis_matrix = monomial_basis_matrix(indices, samples)
        inner_products = (basis_matrix.T*weights).dot(basis_matrix)
        inner_products_exact = np.empty_like(inner_products)
        for ii in range(indices.shape[1]):
            for jj in range(ii, indices.shape[1]):
                inner_products_exact[ii, jj] = monomial_mean_uniform_variables(
                    indices[:, ii:ii+1]+indices[:, jj:jj+1], np.ones((1, 1)))
                inner_products_exact[jj, ii] = inner_products_exact[ii, jj]
        assert np.allclose(inner_products_exact, inner_products)

        # from pyapprox.util.visualization import plot_2d_indices,
        #    plot_3d_indices
        # plot_2d_indices(
        #     indices, other_indices=[sparse_grid.poly_indices,
        #                            double_set_indices])
        # plt.show()


class MultilevelPolynomialModelConfigureVariableTransformation(object):
    def __init__(self, nvars):
        self.nvars = nvars

    def map_from_canonical(self, canonical_samples):
        assert canonical_samples.shape[0] == self.nvars
        samples = canonical_samples.copy()
        samples = samples*2
        return samples

    def num_vars(self):
        return self.nvars


class TestAdaptiveMultiIndexSparseGrid(unittest.TestCase):
    def test_multi_index_sparse_grid(self):
        num_vars = 2
        num_model_levels = 3
        model = MultilevelPolynomialModel(num_model_levels)

        ranges = [2*(-1)**(ii+1) for ii in range(2*num_vars)]
        canonical_ranges = [(-1)**(ii+1) for ii in range(2*num_vars)]
        var_trans = AffineBoundedVariableTransformation(
            canonical_ranges, ranges)
        config_var_trans = \
            MultilevelPolynomialModelConfigureVariableTransformation(1)

        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars+1, num_validation_samples))
        validation_samples[:-1, :] = var_trans.map_from_canonical(
            validation_samples[:-1, :])
        validation_samples[-1, :] = num_model_levels-1
        validation_samples[-1, :] = config_var_trans.map_from_canonical(
            validation_samples[-1:])
        validation_values = model(validation_samples)

        max_level = 5
        max_level_1d = [max_level]*(num_vars+1)
        max_level_1d[-1] = num_model_levels-1
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        refinement_indicator = variance_refinement_indicator
        cost_function = model.cost_function

        sparse_grid = CombinationSparseGrid(num_vars+1)
        sparse_grid.set_function(model, var_trans)
        sparse_grid.set_config_variable_index(num_vars, config_var_trans)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth, cost_function)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        model_level_evals_list = get_num_model_evaluations_from_samples(
            sparse_grid.samples, sparse_grid.num_config_vars)
        model_level_evals = np.asarray(
            model_level_evals_list, dtype=int)[0, :]
        model_ids = np.asarray(model_level_evals_list, dtype=int)[1:, :]
        model_ids = config_var_trans.map_from_canonical(model_ids)
        equivalent_costs, total_costs = get_equivalent_cost(
            cost_function, model_level_evals, model_ids)

        assert np.allclose(
            total_costs, sparse_grid.num_equivalent_function_evaluations)
        assert np.allclose(
            sparse_grid.num_equivalent_function_evaluations/total_costs, 1)

        approx_values = sparse_grid(validation_samples)
        # print np.linalg.norm(approx_values-validation_values)/np.sqrt(
        #     validation_values.shape[0])
        assert np.allclose(approx_values, validation_values)

    def test_online_cost_function(self):
        """
        Test use of work_qoi_index and WorkTracker to determine costs of
        evaluating a model as sparse grid is built
        """
        num_vars = 2
        num_model_levels = 3
        base_model = MultilevelPolynomialModel(
            num_model_levels, return_work=True)
        # TimerModel is hard to test because cost is constantly
        # changing because of variable wall time. So for testing instead use
        # function of polynomial model that just fixes cost for each level of
        # the multilevel model
        timer_model = base_model
        model = WorkTrackingModel(
            timer_model, base_model, 1, enforce_timer_model=False)

        ranges = [2*(-1)**(ii+1) for ii in range(2*num_vars)]
        canonical_ranges = [(-1)**(ii+1) for ii in range(2*num_vars)]
        var_trans = AffineBoundedVariableTransformation(
            canonical_ranges, ranges)
        config_var_trans = \
            MultilevelPolynomialModelConfigureVariableTransformation(1)

        # when computing validation values do not return work
        # or comparison of validation values with approx values will
        # compare matrices of different sizes
        num_validation_samples = 100
        validation_samples = np.random.uniform(
            -1., 1., (num_vars+1, num_validation_samples))
        validation_samples[:-1, :] = var_trans.map_from_canonical(
            validation_samples[:-1, :])
        validation_samples[-1, :] = num_model_levels-1
        validation_samples[-1, :] = config_var_trans.map_from_canonical(
            validation_samples[-1:])
        validation_values = model(validation_samples)

        max_level = 5
        max_level_1d = [max_level]*(num_vars+1)
        max_level_1d[-1] = num_model_levels-1
        max_num_sparse_grid_samples = None
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        refinement_indicator = variance_refinement_indicator
        cost_function = model.cost_function

        sparse_grid = CombinationSparseGrid(num_vars+1)
        sparse_grid.set_function(model, var_trans)
        sparse_grid.set_config_variable_index(num_vars, config_var_trans)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth, cost_function)
        sparse_grid.set_univariate_rules(
            clenshaw_curtis_in_polynomial_order)

        while(not sparse_grid.active_subspace_queue.empty() or
              sparse_grid.subspace_indices.shape[1] == 0):
            sparse_grid.refine()

        model_level_evals_list = get_num_model_evaluations_from_samples(
            sparse_grid.samples, sparse_grid.num_config_vars)
        model_level_evals = np.asarray(
            model_level_evals_list, dtype=int)[0, :]
        model_ids = np.asarray(model_level_evals_list, dtype=int)[1:, :]
        model_ids = config_var_trans.map_from_canonical(model_ids)
        equivalent_costs, total_costs = get_equivalent_cost(
            cost_function, model_level_evals, model_ids)

        # print(total_costs,sparse_grid.num_equivalent_function_evaluations,sparse_grid.num_config_vars)
        assert np.allclose(
            total_costs, sparse_grid.num_equivalent_function_evaluations)
        assert np.allclose(
            sparse_grid.num_equivalent_function_evaluations/total_costs, 1)

        approx_values = sparse_grid(validation_samples)
        # print np.linalg.norm(approx_values-validation_values)/np.sqrt(
        #    validation_values.shape[0])
        assert np.allclose(approx_values, validation_values)

    def test_combination_sparse_grid_setup(self):
        univariate_variables = [stats.uniform(-1, 2)]*2
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        sparse_grid = CombinationSparseGrid(var_trans.num_vars())
        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans)
        max_level_1d = [6]*2
        for ii in range(len(unique_quadrule_indices)):
            for ind in unique_quadrule_indices[ii]:
                max_level_1d[ind] = max(
                    max_level_1d[ind], unique_max_level_1d[ii])
        admissibility_function = partial(
            max_level_admissibility_function, np.inf, [6]*2, 100, 0,
            verbose=False)

        def function(samples):
            return ((samples+1)**5).sum(axis=0)[:, np.newaxis]
        sparse_grid.setup(function, None, variance_refinement_indicator,
                          admissibility_function, growth_rules, quad_rules,
                          var_trans,
                          unique_quadrule_indices=unique_quadrule_indices)
        sparse_grid.build()

        validation_samples = generate_independent_random_samples(
            var_trans.variable, 10)
        vals = sparse_grid(validation_samples)
        validation_values = function(validation_samples)
        assert np.allclose(validation_values, vals)

    def test_insitu_update_sparse_grid_quadrature_rule(self):
        def function(samples):
            return ((samples+1)**5).sum(axis=0)[:, np.newaxis]

        univariate_variables = [stats.uniform(-1, 2)]
        num_vars = len(univariate_variables)
        variable = IndependentMarginalsVariable(
            univariate_variables)
        var_trans = AffineTransform(variable)
        sparse_grid = CombinationSparseGrid(var_trans.num_vars())
        admissibility_function = partial(
            max_level_admissibility_function, np.inf,
            [12]*num_vars, 100, 0, verbose=False)
        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans)
        sparse_grid.setup(
            function, None, isotropic_refinement_indicator,
            admissibility_function, growth_rules, quad_rules,
            var_trans, unique_quadrule_indices=unique_quadrule_indices)

        ninitial_refine_steps = 8
        nsubsequent_refine_steps = 2
        for ii in range(ninitial_refine_steps):
            sparse_grid.refine()

        quadrule_variables = [stats.uniform(-2, 4)]
        var_trans_new = AffineTransform(
            quadrule_variables)
        # map initial points from canonical domain of first range
        # to canconical domain of second range
        initial_points = sparse_grid.samples_1d[0][-1][None, :].copy()
        initial_points = var_trans.map_from_canonical(initial_points)
        initial_points = var_trans_new.map_to_canonical(initial_points)
        # print(initial_points)

        quad_rule = get_univariate_leja_quadrature_rule(
            quadrule_variables[0], growth_rules[0], method='pdf',
            initial_points=initial_points)

        sparse_grid = insitu_update_sparse_grid_quadrature_rule(
            sparse_grid, quadrule_variables)

        for ii in range(nsubsequent_refine_steps):
            sparse_grid.refine()

        samples = quad_rule(sparse_grid.subspace_indices.max())[0]

        assert np.allclose(sparse_grid.samples[0, :], samples)

    def test_piecewise_polynomial_basis(self):
        nvars = 2
        max_level = 2

        variable = IndependentMarginalsVariable([stats.uniform(0, 1)]*nvars)
        var_trans = AffineTransform(variable)
        ranges = variable.get_statistics("interval", 1).flatten()

        w = np.prod([1./(ranges[2*ii+1]-ranges[2*ii])
                     for ii in range(nvars)])
        x, y = sp.Symbol('x'), sp.Symbol('y')
        exact_mean = float(sp.integrate(
            (x+y)**2*w, (x, ranges[0], ranges[1]),
            (y, ranges[2], ranges[3])))

        def function(x): return x.sum(axis=0)[:, None]**2

        refinement_indicator = isotropic_refinement_indicator

        max_level_1d = [max_level]*nvars
        max_num_sparse_grid_samples = 100
        error_tol = None
        admissibility_function = partial(
            max_level_admissibility_function, max_level, max_level_1d,
            max_num_sparse_grid_samples, error_tol)

        basis_type = "quadratic"
        # basis_type = "linear"
        sparse_grid = CombinationSparseGrid(nvars, basis_type)
        sparse_grid.set_refinement_functions(
            refinement_indicator, admissibility_function,
            clenshaw_curtis_rule_growth)
        sparse_grid.set_univariate_rules(
            partial(canonical_univariate_piecewise_polynomial_quad_rule,
                    basis_type))
        sparse_grid.set_function(function, var_trans)
        sparse_grid.build()

        moments = sparse_grid.moments()[:, 0]
        assert np.allclose(moments[0], exact_mean)

        num_validation_samples = 31
        validation_samples = variable.rvs(num_validation_samples)
        values = sparse_grid(validation_samples)
        validation_values = function(validation_samples)
        # print(values.T)
        # print(validation_values.T)
        assert np.allclose(values, validation_values)
        values = sparse_grid.evaluate_using_all_data(validation_samples)
        assert np.allclose(values, validation_values)


if __name__ == "__main__":
    # these functions need to be defined here so pickeling works
    def cost_function(x):
        return 1.

    def function(x):
        return (np.sum(np.exp(x), axis=0)+x[0, :]**3*x[1, :]**3)[:, np.newaxis]

    sparse_grid_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestSparseGrid)
    unittest.TextTestRunner(verbosity=2).run(sparse_grid_test_suite)

    adaptive_sparse_grid_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestAdaptiveSparseGrid)
    unittest.TextTestRunner(verbosity=2).run(adaptive_sparse_grid_test_suite)

    adaptive_multi_index_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestAdaptiveMultiIndexSparseGrid)
    unittest.TextTestRunner(verbosity=2).run(adaptive_multi_index_test_suite)
