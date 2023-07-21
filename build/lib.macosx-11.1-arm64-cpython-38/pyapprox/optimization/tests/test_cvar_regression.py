import unittest
from functools import partial
import numpy as np

from pyapprox.optimization.cvar_regression import (
    smooth_conditional_value_at_risk_gradient,
    smooth_conditional_value_at_risk, smooth_max_function,
    smooth_max_function_first_derivative,
    smooth_max_function_second_derivative,
    smooth_conditional_value_at_risk_composition
)

from pyapprox.variables.risk import value_at_risk

from pyapprox.util.utilities import check_gradients


class TestCVARRegression(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def help_check_smooth_max_function_gradients(self, smoother_type, eps):
        x = np.array([0.01])
        errors = check_gradients(
            partial(smooth_max_function, smoother_type, eps),
            partial(smooth_max_function_first_derivative, smoother_type, eps),
            x[:, np.newaxis])

        errors = check_gradients(
            partial(smooth_max_function_first_derivative, smoother_type, eps),
            partial(smooth_max_function_second_derivative, smoother_type, eps),
            x[:, np.newaxis])
        assert errors.min() < 1e-6

    def test_smooth_max_function_gradients(self):
        smoother_type, eps = 0, 1e-1
        self.help_check_smooth_max_function_gradients(smoother_type, eps)

        smoother_type, eps = 1, 1e-1
        self.help_check_smooth_max_function_gradients(smoother_type, eps)

    def help_check_smooth_conditional_value_at_risk(
            self, smoother_type, eps, alpha):
        samples = np.linspace(-1, 1, 11)
        t = value_at_risk(samples, alpha)[0]
        x0 = np.hstack((samples, t))[:, None]
        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk(
                smoother_type, eps, alpha, xx),
            lambda xx: smooth_conditional_value_at_risk_gradient(
                smoother_type, eps, alpha, xx), x0)
        assert errors.min() < 1e-6

        weights = np.random.uniform(1, 2, samples.shape[0])
        weights /= weights.sum()
        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk(
                smoother_type, eps, alpha, xx, weights),
            lambda xx: smooth_conditional_value_at_risk_gradient(
                smoother_type, eps, alpha, xx, weights), x0)
        assert errors.min() < 1e-6

    def test_smooth_conditional_value_at_risk_gradient(self):
        smoother_type, eps, alpha = 0, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk(
            smoother_type, eps, alpha)

        smoother_type, eps, alpha = 1, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk(
            smoother_type, eps, alpha)

    def help_check_smooth_conditional_value_at_risk_composition_gradient(
            self, smoother_type, eps, alpha, nsamples, nvars):
        samples = np.arange(nsamples*nvars).reshape(nvars, nsamples)
        t = 0.1
        x0 = np.array([2, 3, t])[:, np.newaxis]
        def fun(x): return (np.sum((x*samples)**2, axis=0).T)[:, np.newaxis]
        def jac(x): return 2*(x*samples**2).T

        errors = check_gradients(fun, jac, x0[:2], disp=False)
        assert (errors.min() < 1e-6)

        errors = check_gradients(
            lambda xx: smooth_conditional_value_at_risk_composition(
                smoother_type, eps, alpha, fun, jac, xx),
            True, x0)
        assert errors.min() < 1e-7

    def test_smooth_conditional_value_at_risk_composition_gradient(self):
        nsamples, nvars = 4, 2
        smoother_type, eps, alpha = 0, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk_composition_gradient(
            smoother_type, eps, alpha, nsamples, nvars)

        nsamples, nvars = 10, 2
        smoother_type, eps, alpha = 1, 1e-1, 0.7
        self.help_check_smooth_conditional_value_at_risk_composition_gradient(
            smoother_type, eps, alpha, nsamples, nvars)

if __name__ == "__main__":
    cvar_regression_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestCVARRegression)
    unittest.TextTestRunner(verbosity=2).run(cvar_regression_test_suite)
