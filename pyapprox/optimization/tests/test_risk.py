import unittest

from scipy import stats
import numpy as np

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.optimization.risk import (
    SafetyMarginRiskMeasure,
    AverageValueAtRisk,
    GaussianAnalyticalRiskMeasures,
    ChiSquaredAnalyticalRiskMeasures,
    LogNormalAnalyticalRiskMeasures,
    BetaAnalyticalRiskMeasures,
    DisutilitySSD,
    UtilitySSD,
    KLDivergence,
    HellingerDivergence,
    multivariate_gaussian_kl_divergence,
    CholeskyBasedGaussianExactKLDivergence,
    IndependentGaussianExactKLDivergence,
)


class TestRiskMeasures:
    def setUp(self):
        np.random.seed(1)

    def test_saftey_margins_risk_measure(self):
        bkd = self.get_backend()
        strength = 2.0
        risk_measure = SafetyMarginRiskMeasure(strength, backend=bkd)
        mu, sigma = 1, 2
        rv = stats.norm(mu, sigma)
        nsamples = int(1e6)
        samples = rv.rvs(nsamples)[None, :]
        risk_measure.set_samples(samples)
        assert bkd.allclose(risk_measure(), mu + strength * sigma, rtol=1e-3)

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

    def test_average_value_at_risk(self):
        bkd = self.get_backend()
        mu, sigma, beta = 0, 2, 0.5
        risks = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = risks.AVaR(beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.norm(mu, sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        # print(AVaR(), exact_avar)
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        mu, sigma, beta = 0, 1, 0.5
        risks = LogNormalAnalyticalRiskMeasures(mu, sigma)
        exact_avar = risks.AVaR(beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.lognorm(scale=np.exp(mu), s=sigma)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        ngaussian_vars = 2
        gaussian_samples = np.random.normal(
            mu, sigma, (ngaussian_vars, nsamples)
        )
        risks = ChiSquaredAnalyticalRiskMeasures(ngaussian_vars)
        exact_avar = risks.AVaR(beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        samples = np.sum(gaussian_samples.T**2, axis=1)
        AVaR.set_samples(samples[None, :])
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        a, b, beta = 1, 1, 0.5
        loc, scale = 0, 2.0
        risks = BetaAnalyticalRiskMeasures(a, b, loc, scale)
        exact_avar = risks.AVaR(beta)
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.beta(a, b, loc=loc, scale=scale)
        nsamples = int(1e5)
        AVaR.set_samples(rv.rvs(nsamples)[None, :])
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

    def test_expectation_with_thresholds(self):
        # TODO check disutility and utility are not back to fron
        bkd = self.get_backend()
        mu, sigma, eta = 0, 2, bkd.array([0.25, 0.5])
        risks = LogNormalAnalyticalRiskMeasures(mu, sigma)
        exp_val = risks.disutility_SSD(eta)
        stat = DisutilitySSD(bkd)
        rv = stats.lognorm(scale=np.exp(mu), s=sigma)
        nsamples = int(1e5)
        stat.set_samples(rv.rvs(nsamples)[None, :])
        stat.set_eta(eta)
        assert bkd.allclose(stat(), exp_val, rtol=1e-2)

        exp_val = risks.utility_SSD(eta)
        stat = UtilitySSD(bkd)
        rv = stats.lognorm(scale=np.exp(mu), s=sigma)
        nsamples = int(1e5)
        stat.set_samples(rv.rvs(nsamples)[None, :])
        stat.set_eta(eta)
        assert bkd.allclose(stat(), exp_val, rtol=1e-2)

    def test_compute_f_divergences(self):
        # KL divergence
        mu1, sigma1 = 1, 1
        mu2, sigma2 = 0, 2
        rv1 = stats.norm(mu1, sigma1)
        rv2 = stats.norm(mu2, sigma2)

        # Integrate on [-radius,radius]
        # Note this induces small error by truncating domain..
        # Need to integrate with respect to Lebesque measure
        radius = 10
        quadx, quadw = np.polynomial.legendre.leggauss(400)
        quadx = quadx[None, :] * radius
        quadw = quadw * radius
        div = KLDivergence(rv1.pdf, rv2.pdf, [quadx, quadw])()
        risks = GaussianAnalyticalRiskMeasures(mu1, sigma1)
        true_div = risks.kl_divergence(mu2, sigma2)
        assert np.allclose(div, true_div, rtol=1e-12)

        # Hellinger divergence
        a1, b1, a2, b2 = 1, 1, 2, 3
        rv1, rv2 = stats.beta(a1, b1), stats.beta(a2, b2)
        quadx, quadw = np.polynomial.legendre.leggauss(500)
        quadx = (quadx[None, :] + 1) / 2
        quadw /= 2
        div = HellingerDivergence(rv1.pdf, rv2.pdf, [quadx, quadw])()
        risks = BetaAnalyticalRiskMeasures(a1, b1)
        true_div = risks.hellinger_divergence(a2, b2)
        print(div, true_div)
        assert np.allclose(div, true_div, rtol=1e-10)

    def test_exact_kl_divergence_from_cholesky_factors(self):
        nvars = 3
        mu1 = np.random.uniform(-1, 1, (nvars, 1))
        mu2 = np.random.uniform(-1, 1, (nvars, 1))
        A1 = np.random.uniform(-1, 1, (nvars, nvars))
        A2 = np.random.uniform(-1, 1, (nvars, nvars))
        cov1 = A1.T @ A1
        cov2 = A2.T @ A2
        dense_div = multivariate_gaussian_kl_divergence(mu1, cov1, mu2, cov2)
        chol_div = CholeskyBasedGaussianExactKLDivergence(nvars)
        chol_div.set_left_distribution(mu1, np.linalg.cholesky(cov1))
        chol_div.set_right_distribution(mu2, np.linalg.cholesky(cov2))
        assert np.allclose(chol_div(), dense_div, rtol=1e-12)

    def test_exact_kl_divergence_from_covariance_diagonals(self):
        nvars = 3
        mu1 = np.random.uniform(-1, 1, (nvars, 1))
        mu2 = np.random.uniform(-1, 1, (nvars, 1))
        d1 = np.random.uniform(-1, 1, (nvars, 1)) ** 2
        d2 = np.random.uniform(-1, 1, (nvars, 1)) ** 2
        cov1 = np.diag(d1[:, 0])
        cov2 = np.diag(d2[:, 0])
        dense_div = multivariate_gaussian_kl_divergence(mu1, cov1, mu2, cov2)
        diag_div = IndependentGaussianExactKLDivergence(nvars)
        diag_div.set_left_distribution(mu1, d1)
        diag_div.set_right_distribution(mu2, d2)
        print(diag_div())
        print(dense_div)
        assert np.allclose(diag_div(), dense_div, rtol=1e-12)


class TestNumpyRiskMeasures(TestRiskMeasures, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
