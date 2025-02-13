import unittest

from scipy import stats
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.optimization.risk import AverageValueAtRisk, AnalyticalAVaR


class TestRiskMeasures:
    def setUp(self):
        np.random.seed(1)

    def test_average_value_at_risk_gaussian(self):
        bkd = self.get_backend()
        mu, sigma, beta = 0, 2, 0.5
        exact_avar = AnalyticalAVaR.gaussian(mu, sigma, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.norm(mu, sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        # print(AVaR(), exact_avar)
        assert bkd.allclose(AVaR(), exact_avar, rtol=1e-2)

        mu, sigma, beta = 0, 1, 0.5
        exact_avar = AnalyticalAVaR.lognormal(mu, sigma, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.lognorm(scale=np.exp(mu), s=sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        assert bkd.allclose(AVaR(), exact_avar, rtol=1e-2)

        ngaussian_vars = 2
        gaussian_samples = np.random.normal(
            mu, sigma, (ngaussian_vars, nsamples)
        )
        exact_avar = AnalyticalAVaR.chi_squared(ngaussian_vars, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        samples = np.sum(gaussian_samples.T**2, axis=1)
        AVaR.set_samples(samples[None, :])
        assert bkd.allclose(AVaR(), exact_avar, rtol=1e-2)


class TestNumpyRiskMeasures(TestRiskMeasures, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
