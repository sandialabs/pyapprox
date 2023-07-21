import unittest
import numpy as np
import itertools
from scipy import stats

from pyapprox.benchmarks.genz import GenzFunction
from pyapprox.surrogates.integrate import integrate
from pyapprox.variables.joint import IndependentMarginalsVariable


class TestGenz(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def _check_integrate(self, name, nvars, decay):
        c_factor, w_factor = 1, .5
        fun = GenzFunction()
        fun.set_coefficients(nvars, c_factor, decay, w_factor)
        integral = fun.integrate(name)

        variable = IndependentMarginalsVariable([stats.uniform(0, 1)]*nvars)
        samples, weights = integrate(
            "quasimontecarlo", variable, rule="sobol", nsamples=1e4)
        vals = fun(name, samples)
        qmc_integral = vals.T.dot(weights)
        print(integral, qmc_integral)
        print((qmc_integral-integral)/integral)
        assert np.allclose(qmc_integral, integral, rtol=7e-4)

    def test_integrate(self):
        names = ["oscillatory", "product_peak", "corner_peak", "gaussian",
                 "c0continuous", "discontinuous"]
        nvars = np.arange(2, 7)
        decays = ["none", "quadratic", "quartic", "exp", "sqexp"]
        test_scenarios = itertools.product(*[names, nvars, decays])
        for test_scenario in test_scenarios:
            np.random.seed(1)
            print(test_scenario)
            self._check_integrate(*test_scenario)


if __name__ == "__main__":
    genz_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGenz)
    unittest.TextTestRunner(verbosity=2).run(genz_test_suite)
