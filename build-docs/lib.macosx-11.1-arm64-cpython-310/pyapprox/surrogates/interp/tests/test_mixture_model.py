import unittest
from functools import partial
from scipy import stats
import numpy as np

from pyapprox.surrogates.interp.mixture_model import (
    get_leja_univariate_quadrature_rules_of_beta_mixture, sample_mixture,
    get_mixture_sparse_grid_quadrature_rule,
    get_mixture_tensor_product_gauss_quadrature,
    compute_grammian_of_mixture_models_using_sparse_grid_quadrature
)
from pyapprox.surrogates.orthopoly.quadrature import leja_growth_rule
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion,
    define_poly_options_from_variable_transformation
)
from pyapprox.variables.transforms import (
    define_iid_random_variable_transformation
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices


class TestMixtureModel(unittest.TestCase):
    def test_mixture_model_sparse_grid_quadrature(self):
        num_vars = 2
        level = 5
        rv_params = [[2, 4], [4, 2]]
        rv_params = [[2, 6], [6, 2]]
        # rv_params = [[6,2]]
        num_mixtures = len(rv_params)

        def function(x): return np.array(
            [np.sum(x**2, axis=0), np.sum(x**3, axis=0)+x[0, :]*x[1, :]]).T

        mixture_samplers = []
        for ii in range(num_mixtures):
            def lambda_sampler(a, b, nn): return 2 * \
                np.random.beta(a, b, (num_vars, nn))-1
            # partial is needed to make sure correct alpha and beta parameters
            # are used and not overwritten
            sampler = partial(
                lambda_sampler, rv_params[ii][0], rv_params[ii][1])
            mixture_samplers.append(sampler)
        mc_samples = sample_mixture(mixture_samplers, num_vars, int(1e6))
        mc_integral = function(mc_samples).mean(axis=0)
        # print ('mc',mc_integral)

        leja_basename = None
        mixtures, mixture_univariate_quadrature_rules = \
            get_leja_univariate_quadrature_rules_of_beta_mixture(
                rv_params, leja_growth_rule, leja_basename)
        mixture_univariate_growth_rules = [leja_growth_rule]*num_mixtures
        sg_samples, sg_weights = get_mixture_sparse_grid_quadrature_rule(
            mixture_univariate_quadrature_rules,
            mixture_univariate_growth_rules,
            num_vars, level)
        sg_integral = function(sg_samples).T.dot(sg_weights)
        # print ('sg',sg_integral)
        print('todo: replace with exact analytical integral')
        assert np.allclose(sg_integral, mc_integral, atol=1e-2)

        mixtures, mixture_univariate_quadrature_rules = \
            get_leja_univariate_quadrature_rules_of_beta_mixture(
                rv_params, leja_growth_rule, leja_basename,
                return_weights_for_all_levels=False)
        nquad_samples_1d = leja_growth_rule(level)
        tp_samples, tp_weights = get_mixture_tensor_product_gauss_quadrature(
            mixture_univariate_quadrature_rules, nquad_samples_1d, num_vars)
        tp_integral = function(sg_samples).T.dot(sg_weights)
        # print ('tp',tp_integral)
        assert np.allclose(tp_integral, mc_integral, atol=1e-2)

    def test_compute_grammian_of_mixture_models_using_sparse_grid_quadrature(
            self):
        num_vars = 2
        degree = 3
        # rv_params = [[6,2],[2,6]]
        rv_params = [[1, 1]]
        leja_basename = None
        mixtures, mixture_univariate_quadrature_rules = \
            get_leja_univariate_quadrature_rules_of_beta_mixture(
                rv_params, leja_growth_rule, leja_basename)

        poly = PolynomialChaosExpansion()
        var_trans = define_iid_random_variable_transformation(
            stats.uniform(-1, 2), num_vars)
        poly_opts = define_poly_options_from_variable_transformation(var_trans)
        indices = compute_hyperbolic_indices(num_vars, degree, 1.0)
        poly.configure(poly_opts)
        poly.set_indices(indices)

        num_mixtures = len(rv_params)
        mixture_univariate_growth_rules = [leja_growth_rule]*num_mixtures
        grammian_matrix = \
            compute_grammian_of_mixture_models_using_sparse_grid_quadrature(
                poly.basis_matrix, indices,
                mixture_univariate_quadrature_rules,
                mixture_univariate_growth_rules, num_vars)

        assert (np.all(np.isfinite(grammian_matrix)))

        if num_mixtures == 1:
            II = np.where(abs(grammian_matrix) > 1e-8)
            # check only non-zero inner-products are along diagonal, i.e.
            # for integrals of indices multiplied by themselves
            assert np.allclose(
                II, np.tile(np.arange(indices.shape[1]), (2, 1)))


if __name__ == "__main__":
    mixture_model_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMixtureModel)
    unittest.TextTestRunner(verbosity=2).run(mixture_model_test_suite)
