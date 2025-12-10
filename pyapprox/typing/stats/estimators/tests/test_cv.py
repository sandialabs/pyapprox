"""Tests for CVEstimator."""

import unittest

import numpy as np

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.estimators.cv import CVEstimator
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestCVEstimator(unittest.TestCase):
    """Tests for CVEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_protocol_compliance(self):
        """Test that CVEstimator satisfies EstimatorProtocol."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        self.assertIsInstance(cv, EstimatorProtocol)

    def test_nmodels(self):
        """Test nmodels returns 2."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        self.assertEqual(cv.nmodels(), 2)

    def test_wrong_costs_length(self):
        """Test error for wrong costs length."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)

        # Wrong number of costs
        costs = self.bkd.asarray([10.0, 1.0, 0.1])

        with self.assertRaises(ValueError):
            CVEstimator(stat, costs, self.bkd)

    def test_allocate_samples(self):
        """Test sample allocation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # High correlation: rho = 0.9
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        nsamples = cv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # LF should have more samples than HF
        self.assertGreater(nsamples_np[1], nsamples_np[0])

        # Total cost should be approximately target
        total_cost = cv.total_cost()
        self.assertLessEqual(total_cost, 100.0 * 1.1)  # Allow 10% tolerance

    def test_weights(self):
        """Test optimal weight computation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # Cov(Q0, Q1) = 0.9, Var(Q1) = 1.0
        # eta = 0.9 / 1.0 = 0.9
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        weights = cv.weights()
        np.testing.assert_allclose(
            self.bkd.to_numpy(weights)[0], 0.9, rtol=1e-10
        )

    def test_generate_samples(self):
        """Test sample generation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        np.random.seed(42)
        def rvs(n):
            return self.bkd.asarray(np.random.randn(n, 2))

        samples = cv.generate_samples_per_model(rvs)

        self.assertEqual(len(samples), 2)

        nsamples = cv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)

        # HF samples
        self.assertEqual(samples[0].shape[0], int(nsamples_np[0]))
        # LF samples
        self.assertEqual(samples[1].shape[0], int(nsamples_np[1]))

        # First nhf samples of LF should match HF samples (shared)
        np.testing.assert_array_equal(
            self.bkd.to_numpy(samples[0]),
            self.bkd.to_numpy(samples[1][:int(nsamples_np[0])])
        )

    def test_estimate_perfect_correlation(self):
        """Test CV estimate with perfect correlation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        # Perfect correlation: LF = HF
        cov = self.bkd.asarray([[1.0, 1.0], [1.0, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        nsamples = cv.nsamples_per_model()
        nsamples_np = self.bkd.to_numpy(nsamples)
        nhf = int(nsamples_np[0])
        nlf = int(nsamples_np[1])

        # Create perfectly correlated values
        hf_values = self.bkd.asarray(np.arange(1, nhf + 1).reshape(-1, 1).astype(float))

        # LF has same first nhf values, then more
        lf_vals = np.arange(1, nlf + 1).reshape(-1, 1).astype(float)
        lf_values = self.bkd.asarray(lf_vals)

        estimate = cv([hf_values, lf_values])

        # Should be close to mean of LF (which has more samples)
        expected = np.mean(lf_vals)
        np.testing.assert_allclose(
            self.bkd.to_numpy(estimate)[0], expected, rtol=0.1
        )

    def test_variance_reduction_high_correlation(self):
        """Test variance reduction with high correlation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        rho = 0.9
        cov = self.bkd.asarray([[1.0, rho], [rho, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)
        cv.allocate_samples(target_cost=100.0)

        var_red = cv.variance_reduction()

        # Should be approximately 1 - rho^2 = 1 - 0.81 = 0.19
        expected = 1 - rho**2
        np.testing.assert_allclose(var_red, expected, rtol=1e-10)

    def test_repr(self):
        """Test string representation."""
        stat = MultiOutputMean(nqoi=1, bkd=self.bkd)
        cov = self.bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = self.bkd.asarray([10.0, 1.0])

        cv = CVEstimator(stat, costs, self.bkd)

        # Before allocation
        repr_str = repr(cv)
        self.assertIn("CVEstimator", repr_str)
        self.assertIn("not allocated", repr_str)

        # After allocation
        cv.allocate_samples(target_cost=100.0)
        repr_str = repr(cv)
        self.assertIn("[", repr_str)


class TestCVEstimatorNumerical(unittest.TestCase):
    """Numerical tests for CVEstimator."""

    def setUp(self):
        self.bkd = NumpyBkd()

    def test_variance_reduction_monte_carlo(self):
        """Test that CV actually reduces variance."""
        np.random.seed(42)

        # True mean
        true_mean = 0.0

        # Covariance structure
        rho = 0.95
        cov = self.bkd.asarray([[1.0, rho], [rho, 1.0]])

        # Run many replications
        n_rep = 1000
        mc_estimates = []
        cv_estimates = []

        for _ in range(n_rep):
            # Generate correlated samples
            n = 50
            z1 = np.random.randn(n)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(n)

            # MC estimate (HF only)
            nhf = 5
            mc_est = np.mean(z1[:nhf])
            mc_estimates.append(mc_est)

            # CV estimate
            eta = rho
            hf_vals = z1[:nhf]
            lf_vals = z2[:n]
            Q0 = np.mean(hf_vals)
            Q1 = np.mean(lf_vals[:nhf])
            mu1 = np.mean(lf_vals)
            cv_est = Q0 + eta * (mu1 - Q1)
            cv_estimates.append(cv_est)

        mc_var = np.var(mc_estimates)
        cv_var = np.var(cv_estimates)

        # CV should have lower variance
        self.assertLess(cv_var, mc_var)

        # Variance ratio should be approximately 1 - rho^2
        expected_ratio = 1 - rho**2
        actual_ratio = cv_var / mc_var
        # Allow generous tolerance due to Monte Carlo variability
        self.assertLess(actual_ratio, expected_ratio * 3)


if __name__ == "__main__":
    unittest.main()
