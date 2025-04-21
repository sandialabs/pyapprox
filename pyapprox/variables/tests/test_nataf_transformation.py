import numpy as np
from scipy import stats
from functools import partial
import unittest

from pyapprox.util.misc import (
    get_tensor_product_quadrature_rule,
    correlation_to_covariance,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.transforms import NatafTransform
from pyapprox.variables._nataf import generate_x_samples_using_gaussian_copula
from pyapprox.variables.marginals import ContinuousScipyMarginal


class TestNatafTransform:
    def setUp(self):
        np.random.seed(1)

    def test_independent_gaussian(self):
        bkd = self.get_backend()
        nvars = 2
        nsamples = 3
        x_marginals = [ContinuousScipyMarginal(stats.norm(0, 1), bkd)] * nvars
        x_covariance = bkd.eye(nvars)
        x_samples = bkd.asarray(np.random.normal(0.0, 1.0, (nvars, nsamples)))
        trans = NatafTransform(x_marginals, x_covariance)
        u_samples = trans.map_to_canonical(x_samples)
        assert np.allclose(u_samples, x_samples)

    def test_correlated_gaussian(self):
        bkd = self.get_backend()
        nvars = 2
        nsamples = 10
        x_marginals = [ContinuousScipyMarginal(stats.norm(0, 1), bkd)] * nvars
        x_correlation = np.array([[1, 0.5], [0.5, 1]])
        x_covariance = x_correlation.copy()  # because variances are 1.0
        x_covariance_chol_factor = np.linalg.cholesky(x_covariance)
        iid_x_samples = np.random.normal(0.0, 1.0, (nvars, nsamples))
        x_samples = x_covariance_chol_factor @ iid_x_samples
        trans = NatafTransform(x_marginals, x_covariance)
        u_samples = trans.map_to_canonical(x_samples)
        assert np.allclose(u_samples, iid_x_samples)

    def test_correlated_gamma(self):
        bkd = self.get_backend()
        nvars = 2
        x_marginals = [
            ContinuousScipyMarginal(stats.gamma(a=2, scale=3), bkd)
        ] * nvars
        x_correlation = np.array([[1, 0.7], [0.7, 1]])

        x_covariance = correlation_to_covariance(
            x_correlation, bkd.array([m.std() for m in x_marginals])
        )
        trans = NatafTransform(x_marginals, x_covariance)
        # from Li HongShuang et al.
        true_z_correlation = np.asarray([[1.0, 0.7207], [0.7207, 1.0]])
        assert np.allclose(trans._z_correlation, true_z_correlation, atol=1e-4)

    def test_correlated_beta(self):
        bkd = self.get_backend()
        nvars = 2
        alpha_stat = 2
        beta_stat = 5
        bisection_opts = {"tol": 1e-10, "max_iterations": 100}
        x_marginals = [
            ContinuousScipyMarginal(stats.beta(a=alpha_stat, b=beta_stat), bkd)
        ] * nvars
        x_correlation = np.array([[1, 0.7], [0.7, 1]])

        x_covariance = correlation_to_covariance(
            x_correlation, bkd.array([m.std() for m in x_marginals])
        )
        trans = NatafTransform(x_marginals, x_covariance)

        assert np.allclose(
            trans._z_correlation[0, 1], trans._z_correlation[1, 0]
        )

        x_correlation_recovered = trans.z_correlation_to_x_correlation(
            trans._z_correlation
        )
        assert np.allclose(x_correlation, x_correlation_recovered)

        # all variances are the same so
        # true_x_covariance  = x_correlation.copy()*x_marginal_stdevs[0]**2
        true_x_covariance = correlation_to_covariance(
            x_correlation, bkd.array([m.std() for m in x_marginals])
        )

        def univariate_quad_rule(n):
            x, w = np.polynomial.legendre.leggauss(n)
            x = (x + 1.0) / 2.0
            w /= 2.0
            return x, w

        x, w = get_tensor_product_quadrature_rule(
            50, nvars, univariate_quad_rule
        )
        target_density = trans.pdf
        assert np.allclose(target_density(x) @ w, 1.0)

        # test covariance of computed by aplying quadrature to joint density
        mean = (x * target_density(x)) @ w
        x_covariance = np.empty((nvars, nvars))
        x_covariance[0, 0] = (x[0, :] ** 2 * target_density(x)) @ w - mean[
            0
        ] ** 2
        x_covariance[1, 1] = (x[1, :] ** 2 * target_density(x)) @ w - mean[
            1
        ] ** 2
        x_covariance[0, 1] = (
            x[0, :] * x[1, :] * target_density(x)
        ) @ w - mean[0] * mean[1]
        x_covariance[1, 0] = x_covariance[0, 1]
        # error is influenced by bisection_opts['tol']
        assert np.allclose(
            x_covariance, true_x_covariance, atol=bisection_opts["tol"]
        )

        # test samples generated using Gaussian copula are correct
        nsamples = 10000
        x_samples, true_u_samples = generate_x_samples_using_gaussian_copula(
            nvars, trans._z_correlation, x_marginals, nsamples
        )

        x_sample_covariance = np.cov(x_samples)
        assert np.allclose(true_x_covariance, x_sample_covariance, atol=1e-2)

        trans = NatafTransform(x_marginals, true_x_covariance)
        u_samples = trans.map_to_canonical(x_samples)
        assert np.allclose(u_samples, true_u_samples)
        trans_samples = trans.map_from_canonical(u_samples)
        assert np.allclose(x_samples, trans_samples)


class TestNumpyNatafTransform(TestNatafTransform, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
