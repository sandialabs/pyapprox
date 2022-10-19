import unittest
from scipy import stats
import numpy as np

from pyapprox.surrogates.integrate import integrate
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.surrogates.interp.monomial import (
    evaluate_monomial, monomial_mean_uniform_variables,
    monomial_mean_gaussian_variables)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices


class TestIntegrate(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_monomial_integration(self, level, method, marginals, opts={},
                                    tol=1e-12, var_type="uniform"):
        variable = IndependentMarginalsVariable(marginals)
        samples, weights = integrate(method, variable, **opts)
        indices = compute_hyperbolic_indices(variable.num_vars(), level)
        coeffs = np.random.normal(0, 1, (indices.shape[1], 1))
        vals = evaluate_monomial(indices, coeffs, samples)
        integral = vals.T.dot(weights)
        if var_type == "uniform":
            exact_integral = monomial_mean_uniform_variables(indices, coeffs)
        else:
            exact_integral = monomial_mean_gaussian_variables(indices, coeffs)
        print(integral)
        print(integral-exact_integral)
        assert np.allclose(integral, exact_integral, rtol=tol)

    def test_uniform_integration(self):
        for nvars in [2, 3]:
            test_scenarios = [
                [1, "tensor_product", [stats.uniform(-1, 2)]*nvars,
                 {"rule": "linear", "levels": 40, "growth": "two_point"}],
                [2, "tensor_product", [stats.uniform(-1, 2)]*nvars,
                 {"rule": "quadratic", "levels": 2, "growth": "one_point"}],
                [2, "tensor_product", [stats.uniform(-1, 2)]*nvars],
                [2, "sparse_grid", [stats.uniform(-1, 2)]*nvars,
                 {"growth_rule": "clenshaw_curtis"}],
                [2, "tensor_product", [stats.uniform(-1, 2)]*nvars,
                 {"growth_rule": "clenshaw_curtis", "rule": "leja",
                  "levels": 2}],
            ]
            for test_scenario in test_scenarios:
                self._check_monomial_integration(*test_scenario)

    def test_gaussian_integration(self):
        for nvars in [2, 3]:
            test_scenarios = [
                [1, "tensor_product", [stats.norm(0, 1)]*nvars,
                 {"rule": "linear", "levels": 40, "growth": "two_point"}],
                [2, "tensor_product", [stats.norm(0, 1)]*nvars,
                 {"rule": "quadratic", "levels": 2, "growth": "one_point"}],
                [2, "tensor_product", [stats.norm(0, 1)]*nvars],
                [2, "sparse_grid", [stats.norm(0, 1)]*nvars,
                 {"growth_rule": "clenshaw_curtis"}],
                [2, "tensor_product", [stats.norm(0, 1)]*nvars,
                 {"growth_rule": "clenshaw_curtis", "rule": "leja",
                  "levels": 2}],
            ]
            for test_scenario in test_scenarios[2:3]:
                self._check_monomial_integration(
                    *test_scenario, var_type="gaussian")


if __name__ == "__main__":
    integrate_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestIntegrate)
    unittest.TextTestRunner(verbosity=2).run(integrate_test_suite)
