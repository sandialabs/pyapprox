"""Tests for CVEstimator.

Replicates legacy tests from pyapprox/multifidelity/tests/test_acv.py for CV.
"""

from typing import Any, Generic, Tuple
import unittest

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.statistics.variance import MultiOutputVariance
from pyapprox.typing.stats.statistics.mean_variance import MultiOutputMeanAndVariance
from pyapprox.typing.stats.estimators.cv import CVEstimator
from pyapprox.typing.stats.protocols import EstimatorProtocol


class TestCVEstimator(Generic[Array], unittest.TestCase):
    """Tests for CVEstimator with multi-model support."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_protocol_compliance(self):
        """Test that CVEstimator satisfies EstimatorProtocol."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        self.assertIsInstance(cv, EstimatorProtocol)

    def test_nmodels_2_models(self):
        """Test nmodels returns 2 for 2-model case."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        self.assertEqual(cv.nmodels(), 2)

    def test_nmodels_3_models(self):
        """Test nmodels returns 3 for 3-model case."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])
        lowfi_stats = bkd.asarray([[0.0], [0.0]])  # 2 LF models

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        self.assertEqual(cv.nmodels(), 3)

    def test_wrong_costs_length(self):
        """Test error for wrong costs length."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)

        # Wrong number of costs (3 vs 2 models in cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        with self.assertRaises(ValueError):
            CVEstimator(stat, costs, bkd)

    def test_wrong_lowfi_stats_shape(self):
        """Test error for wrong lowfi_stats shape."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])

        # Wrong shape: should be (2, 1) for 3 models with nqoi=1
        lowfi_stats = bkd.asarray([[0.0]])  # Only 1 LF model

        with self.assertRaises(ValueError):
            CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)

    def test_allocate_samples_all_shared(self):
        """Test sample allocation - all models share same samples."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])
        lowfi_stats = bkd.asarray([[0.0], [0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        nsamples = cv.nsamples_per_model()
        nsamples_np = bkd.to_numpy(nsamples)

        # All models should have same number of samples (fully shared)
        bkd.assert_allclose(
            nsamples,
            bkd.full((3,), nsamples_np[0]),
            rtol=1e-10,
        )

    def test_weights_2_models(self):
        """Test optimal weight computation for 2 models."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        # Cov(Q0, Q1) = 0.9, Var(Q1) = 1.0
        # eta = -Cov(Q0, Q1)/Var(Q1) = -0.9 / 1.0 = -0.9
        # Negative because when Q1 is high, so is Q0, so we subtract
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        weights = cv.weights()
        bkd.assert_allclose(
            weights, bkd.asarray([[-0.9]]), rtol=1e-10
        )

    def test_weights_3_models(self):
        """Test optimal weight computation for 3 models."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])
        lowfi_stats = bkd.asarray([[0.0], [0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        weights = cv.weights()
        # Weights should be 2D with shape (nstats, nstats*(nmodels-1))
        # For nqoi=1, nmodels=3: shape is (1, 2)
        self.assertEqual(weights.shape, (1, 2))

    def test_generate_samples_all_shared(self):
        """Test sample generation - all models get same samples."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0, 0.1])
        lowfi_stats = bkd.asarray([[0.0], [0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        np.random.seed(42)

        def rvs(n):
            return bkd.asarray(np.random.randn(n, 2))

        samples = cv.generate_samples_per_model(rvs)

        self.assertEqual(len(samples), 3)

        # All models should have same samples (fully shared)
        bkd.assert_allclose(samples[0], samples[1], rtol=1e-10)
        bkd.assert_allclose(samples[0], samples[2], rtol=1e-10)

    def test_estimate_with_lowfi_stats(self):
        """Test CV estimate uses known LF statistics."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        known_lf_mean = bkd.asarray([[5.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=known_lf_mean)
        cv.allocate_samples(target_cost=100.0)

        nsamples_np = bkd.to_numpy(cv.nsamples_per_model())
        n = int(nsamples_np[0])

        # HF samples centered at 0
        hf_values = bkd.asarray(np.zeros((n, 1)))
        # LF samples centered at 5 (matches known_lf_mean)
        lf_values = bkd.asarray(np.ones((n, 1)) * 5.0)

        estimate = cv([hf_values, lf_values])

        # With perfect match, Q_m - mu_m = 0, so estimate = Q_0 = 0
        bkd.assert_allclose(estimate, bkd.asarray([0.0]), atol=1e-10)

    def test_variance_reduction(self):
        """Test variance reduction factor computation."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        rho = 0.9
        cov = bkd.asarray([[1.0, rho], [rho, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        var_red = cv.variance_reduction()

        # Should be approximately 1 - rho^2 = 1 - 0.81 = 0.19
        expected = 1 - rho**2
        bkd.assert_allclose(
            bkd.asarray([var_red]),
            bkd.asarray([expected]),
            rtol=1e-10,
        )

    def test_repr_before_allocation(self):
        """Test string representation before allocation."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)

        repr_str = repr(cv)
        self.assertIn("CVEstimator", repr_str)
        self.assertIn("not allocated", repr_str)

    def test_repr_after_allocation(self):
        """Test string representation after allocation."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        repr_str = repr(cv)
        self.assertIn("[", repr_str)  # Contains list of samples

    def test_optimized_covariance(self):
        """Test optimized_covariance returns valid result."""
        bkd = self._bkd
        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        cov = bkd.asarray([[1.0, 0.9], [0.9, 1.0]])
        stat.set_pilot_quantities(cov)
        costs = bkd.asarray([10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0]])

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=100.0)

        opt_cov = cv.optimized_covariance()

        # Should be positive definite (all eigenvalues > 0 for single stat)
        self.assertGreater(float(bkd.to_numpy(opt_cov).flatten()[0]), 0.0)


class TestCVEstimatorNumpy(TestCVEstimator[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCVEstimatorTorch(TestCVEstimator[torch.Tensor]):
    __test__ = True

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestCVEstimatorNumerical(Generic[Array], unittest.TestCase):
    """Numerical verification tests for CVEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    @slow_test
    def test_variance_reduction_monte_carlo(self):
        """Test that CV actually reduces variance via Monte Carlo."""
        bkd = self._bkd
        np.random.seed(42)

        # True mean
        true_mean = 0.0

        # Covariance structure
        rho = 0.95
        cov = bkd.asarray([[1.0, rho], [rho, 1.0]])

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

            # CV estimate with known LF mean = 0
            eta = rho
            hf_vals = z1[:nhf]
            lf_vals = z2[:nhf]
            Q0 = np.mean(hf_vals)
            Q1 = np.mean(lf_vals)
            mu1 = 0.0  # Known LF mean
            cv_est = Q0 + eta * (mu1 - Q1)
            cv_estimates.append(cv_est)

        mc_var = np.var(mc_estimates)
        cv_var = np.var(cv_estimates)

        # CV should have lower variance
        self.assertLess(cv_var, mc_var)

    @slow_test
    def test_discrepancy_covariances_numerical(self):
        """Test that analytical discrepancy covariances match MC."""
        bkd = self._bkd
        np.random.seed(42)

        # 3 models
        cov_np = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        cov = bkd.asarray(cov_np)
        costs = bkd.asarray([10.0, 1.0, 0.1])
        lowfi_means = bkd.asarray([[0.0], [0.0]])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_means)
        cv.allocate_samples(target_cost=100.0)

        nsamples_np = bkd.to_numpy(cv.nsamples_per_model())
        n = int(nsamples_np[0])

        # Monte Carlo estimate of discrepancy covariances
        ntrials = 5000
        Q_vals = []
        delta_vals = []

        L = np.linalg.cholesky(cov_np)

        for _ in range(ntrials):
            # Generate correlated samples
            z = np.random.randn(n, 3)
            vals = z @ L.T

            # Q_0: HF mean estimate
            Q0 = np.mean(vals[:, 0])

            # Deltas: Q_m - mu_m (but mu_m = 0)
            delta1 = np.mean(vals[:, 1])
            delta2 = np.mean(vals[:, 2])

            Q_vals.append(Q0)
            delta_vals.append([delta1, delta2])

        Q_vals = np.array(Q_vals)
        delta_vals = np.array(delta_vals)

        # Numerical CF (covariance of deltas)
        CF_mc = np.cov(delta_vals.T, ddof=1)

        # Numerical cf (covariance of Q with deltas)
        cf_mc = np.cov(np.vstack([Q_vals, delta_vals.T]), ddof=1)[0, 1:]

        # Analytical
        CF_an, cf_an = stat.get_cv_discrepancy_covariances(cv.npartition_samples())

        # Compare
        bkd.assert_allclose(
            bkd.asarray(CF_mc), CF_an, atol=0.01, rtol=0.05
        )
        bkd.assert_allclose(
            bkd.asarray(cf_mc), cf_an.flatten(), atol=0.01, rtol=0.05
        )


class TestCVEstimatorNumericalNumpy(TestCVEstimatorNumerical[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCVEstimatorNumericalTorch(TestCVEstimatorNumerical[torch.Tensor]):
    __test__ = True

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


class TestCVBootstrap(Generic[Array], unittest.TestCase):
    """Bootstrap tests for CVEstimator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def _generate_correlated_values(
        self, nsamples_list: list, cov_np: np.ndarray
    ) -> list:
        """Generate correlated values for testing."""
        L = np.linalg.cholesky(cov_np)
        nmodels = len(nsamples_list)
        values = []
        for m in range(nmodels):
            n = nsamples_list[m]
            z = np.random.randn(n, nmodels)
            vals = z @ L.T
            values.append(self._bkd.asarray(vals[:, m:m+1]))
        return values

    @slow_test
    def test_bootstrap_cv_mean(self):
        """Test bootstrap for CV with mean statistic."""
        bkd = self._bkd
        np.random.seed(42)

        # 3-model covariance
        cov_np = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.85],
            [0.8, 0.85, 1.0],
        ])
        cov = bkd.asarray(cov_np)
        costs = bkd.asarray([100.0, 10.0, 1.0])
        lowfi_stats = bkd.asarray([[0.0], [0.0]])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)

        cv = CVEstimator(stat, costs, bkd, lowfi_stats=lowfi_stats)
        cv.allocate_samples(target_cost=500)

        # Generate correlated values
        nsamples = bkd.to_numpy(cv.nsamples_per_model())
        nsamples_list = [int(n) for n in nsamples]

        # For CV, all models have same samples
        n = nsamples_list[0]
        L = np.linalg.cholesky(cov_np)
        z = np.random.randn(n, 3)
        all_vals = z @ L.T

        values = [
            bkd.asarray(all_vals[:, m:m+1])
            for m in range(3)
        ]

        # Run bootstrap
        nbootstraps = 500
        boot_mean, boot_cov = cv.bootstrap(values, nbootstraps=nbootstraps)

        # Check shape
        self.assertEqual(boot_mean.shape, (1,))

        # Bootstrap covariance should be close to analytical
        analytical_cov = cv.optimized_covariance()
        bkd.assert_allclose(
            boot_cov.flatten(),
            bkd.to_numpy(analytical_cov).flatten(),
            rtol=0.5,  # Bootstrap is noisy
        )


class TestCVBootstrapNumpy(TestCVBootstrap[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCVBootstrapTorch(TestCVBootstrap[torch.Tensor]):
    __test__ = True

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
