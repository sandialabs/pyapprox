import unittest

from scipy import stats
import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.optimization.risk import AverageValueAtRisk, AnalyticalAVaR


class TestRiskMeasures:
    def setUp(self):
        np.random.seed(1)

    def test_average_value_at_risk_discrete_measure(self):
        # Psi(VaR) = prob samples do not exceed or are equal to VaR
        # lambda = (Psi(VaR) - beta) / (1 - beta)
        # AVaR- expected values are equal to or exceed VaR
        # AVaR+ expected values of values strictly exceeding VaR
        # AVaR = weighted average of VaR and AVaR+
        # AVaR = lambda * VaR + (1-lambda * AVaR+)

        bkd = self.get_backend()
        nsamples = 6
        mu, sigma = 0, 2
        rv = stats.norm(mu, sigma)
        samples = rv.rvs(nsamples)[None, :]

        # lambda = 0, VaR<AVaR-<AVaR=AVaR+
        beta = 4 / 6
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        AVaR.set_samples(samples)
        assert bkd.allclose(
            0.5 * (AVaR._samples[4] + AVaR._samples[5]), AVaR()[0]
        )
        assert bkd.allclose(AVaR()[1], AVaR._samples[3])

        # lambda > 0, VaR<AVaR-<AVaR<AVaR+
        # AVaR = 1/5 VaR + 2/5 * AVaR+
        beta = 7 / 12
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        AVaR.set_samples(samples)
        assert bkd.allclose(
            AVaR()[0],
            0.2 * AVaR._samples[3]
            + 0.4 * (AVaR._samples[4] + AVaR._samples[5]),
        )
        assert bkd.allclose(AVaR()[1], AVaR._samples[3])
        assert bkd.allclose(AVaR.optimize()[0], AVaR()[0])
        assert bkd.allclose(AVaR.optimize()[1], AVaR()[1])

        # VaR is the last sample
        beta = 11 / 12
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        AVaR.set_samples(samples)
        assert bkd.allclose(AVaR()[0], AVaR._samples[5])
        assert bkd.allclose(AVaR()[1], AVaR._samples[5])

    def test_average_value_at_risk_gaussian(self):
        bkd = self.get_backend()
        mu, sigma, beta = 0, 2, 0.5
        exact_avar = AnalyticalAVaR.gaussian(mu, sigma, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.norm(mu, sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        # print(AVaR(), exact_avar)
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        mu, sigma, beta = 0, 1, 0.5
        exact_avar = AnalyticalAVaR.lognormal(mu, sigma, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.lognorm(scale=np.exp(mu), s=sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        ngaussian_vars = 2
        gaussian_samples = np.random.normal(
            mu, sigma, (ngaussian_vars, nsamples)
        )
        exact_avar = AnalyticalAVaR.chi_squared(ngaussian_vars, beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        samples = np.sum(gaussian_samples.T**2, axis=1)
        AVaR.set_samples(samples[None, :])
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)


class TestNumpyRiskMeasures(TestRiskMeasures, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
