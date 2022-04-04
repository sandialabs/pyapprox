import unittest
import numpy as np

from pyapprox.optimization.quantile_regression import solve_quantile_regression


try:
    import cvxopt
    cvxopt_missing = False
except ImportError:
    cvxopt_missing = True
skipcvxopttest = unittest.skipIf(
    cvxopt_missing, reason="cvxopt package missing")


class TestQuantileRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    @skipcvxopttest
    def test_quantile_regression(self):
        np.random.seed(1)
        nbasis = 20

        def func(x):
            return (1+x-x**2+x**3).T
        samples = np.random.uniform(-1, 1, (1, 201))
        values = func(samples)

        def eval_basis_matrix(x):
            return (x**np.arange(nbasis)[:, None]).T
        tau = 0.75
        quantile_coef = solve_quantile_regression(
            tau, samples, values, eval_basis_matrix)
        true_coef = np.zeros((nbasis))
        true_coef[:4] = [1, 1, -1, 1]
        assert np.allclose(quantile_coef[:, 0], true_coef)


if __name__ == "__main__":
    quantile_regression_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestQuantileRegression)
    unittest.TextTestRunner(verbosity=2).run(quantile_regression_test_suite)
