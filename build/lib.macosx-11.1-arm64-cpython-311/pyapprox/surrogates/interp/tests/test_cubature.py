import unittest
import numpy as np

from pyapprox.surrogates.interp.cubature import get_cubature_rule
from pyapprox.surrogates.interp.monomial import (
    monomial_mean_uniform_variables, evaluate_monomial)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices


class TestCubature(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_cubature_rule(self, nvars, degree):
        indices = compute_hyperbolic_indices(nvars, degree)
        coeffs = np.random.normal(0, 1, (indices.shape[1], 1))
        exact_integral = monomial_mean_uniform_variables(indices, coeffs)
        x_quad, w_quad = get_cubature_rule(nvars, degree)
        vals = evaluate_monomial(indices, coeffs, x_quad)
        integral = vals.T.dot(w_quad)

        # print((exact_integral, integral))
        assert np.allclose(exact_integral, integral)

    def test_cubature_rules(self):
        for degree in [2, 3, 5]:
            for nvars in np.arange(2, 10):
                self._check_cubature_rule(nvars, degree)


if __name__ == "__main__":
    cubature_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestCubature)
    unittest.TextTestRunner(verbosity=2).run(cubature_test_suite)
