import unittest
from pyapprox.quantile_regression import *
from pyapprox.risk_measures import value_at_risk
from pyapprox.optimization import check_gradients
from scipy.special import erf, erfinv, factorial
from scipy import stats

class TestQuantileRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        
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
