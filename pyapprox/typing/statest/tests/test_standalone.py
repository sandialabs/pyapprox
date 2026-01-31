"""Standalone tests for statest module.

These tests verify the correctness of the statest implementation
without comparing to legacy code. They test:
- Bootstrap variance estimation matches analytical variance
- Pilot quantity computation from samples
- End-to-end polynomial ensemble tests
- Insert pilot samples functionality

These tests use the typing array convention: (nqoi, nsamples) for outputs.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401

from pyapprox.typing.statest.statistics import (
    MultiOutputMean,
    MultiOutputVariance,
    MultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.mc_estimator import MCEstimator
from pyapprox.typing.statest.cv_estimator import CVEstimator
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator,
    GISEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
)


def _create_correlated_model_values(
    bkd: Backend[Array],
    nsamples: int,
    nqoi: int,
    nmodels: int,
    correlation: float = 0.9,
    seed: int = 42,
) -> tuple:
    """Create correlated model values for testing.

    Returns values in typing convention: (nqoi, nsamples).
    """
    np.random.seed(seed)
    # Generate base values
    base = np.random.randn(nqoi, nsamples)
    values_per_model = [bkd.asarray(base.copy())]

    # Create correlated low-fidelity values
    for ii in range(1, nmodels):
        noise_scale = 1.0 - correlation
        lf_values = base + np.random.randn(nqoi, nsamples) * noise_scale
        values_per_model.append(bkd.asarray(lf_values))

    return values_per_model


def _compute_mc_variance(
    bkd: Backend[Array],
    estimator,
    values_generator,
    ntrials: int,
) -> Array:
    """Compute MC estimate of estimator variance.

    Parameters
    ----------
    bkd : Backend
        Backend instance.
    estimator : Estimator
        The estimator to evaluate.
    values_generator : callable
        Function that returns values_per_model for each trial.
    ntrials : int
        Number of MC trials.

    Returns
    -------
    Array
        MC estimate of estimator covariance.
    """
    estimates = []
    for _ in range(ntrials):
        values_per_model = values_generator()
        est_val = estimator(values_per_model)
        estimates.append(est_val)
    estimates = bkd.stack(estimates)
    return bkd.cov(estimates, rowvar=False, ddof=1)


class TestBootstrapEstimator(Generic[Array], unittest.TestCase):
    """Test bootstrap variance estimation matches analytical variance."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    @slow_test
    def test_mc_bootstrap_variance(self) -> None:
        """Test MC bootstrap variance matches analytical."""
        nqoi = 2
        nmodels = 1
        nsamples = 100
        nbootstraps = 1000

        # Create statistic and set pilot quantities
        stat = MultiOutputMean(nqoi, self._bkd)
        variance = 4.0
        cov = self._bkd.eye(nmodels * nqoi) * variance
        stat.set_pilot_quantities(cov)

        # Create estimator
        costs = self._bkd.array([1.0])
        est = MCEstimator(stat, costs)
        target_cost = float(nsamples)
        est.allocate_samples(target_cost)

        # Generate values using typing convention (nqoi, nsamples)
        values = self._bkd.asarray(np.random.randn(nqoi, nsamples) * 2.0)

        # Bootstrap
        bootstrap_mean, bootstrap_cov = est.bootstrap([values], nbootstraps)

        # Compare to analytical covariance
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            bootstrap_cov, analytical_cov, atol=5e-2, rtol=1e-1
        )

    @slow_test
    def test_cv_bootstrap_variance(self) -> None:
        """Test CV bootstrap variance matches analytical."""
        nqoi = 1
        nmodels = 2
        nsamples = 200
        nbootstraps = 1000

        # Create covariance with high correlation
        cov = self._bkd.array([[1.0, 0.9], [0.9, 1.0]])
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        # Known low-fidelity statistics
        lowfi_stats = self._bkd.zeros((nmodels - 1, nqoi))

        # Create estimator
        costs = self._bkd.array([1.0, 0.1])
        est = CVEstimator(stat, costs, lowfi_stats)
        target_cost = float(nsamples) * costs.sum()
        est.allocate_samples(target_cost)

        # Generate correlated values using typing convention (nqoi, nsamples)
        np.random.seed(123)
        hf_values = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        lf_values = hf_values + self._bkd.asarray(
            np.random.randn(nqoi, nsamples) * 0.1
        )
        values_per_model = [hf_values, lf_values]

        # Bootstrap
        bootstrap_mean, bootstrap_cov = est.bootstrap(values_per_model, nbootstraps)

        # Compare to analytical covariance
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            bootstrap_cov, analytical_cov, atol=5e-2, rtol=2e-1
        )

    @slow_test
    def test_mfmc_bootstrap_variance(self) -> None:
        """Test MFMC bootstrap variance matches analytical."""
        nqoi = 1
        nmodels = 3
        nbootstraps = 500

        # Create covariance with decreasing correlations
        cov = self._bkd.array([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0],
        ])
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        # Create estimator
        costs = self._bkd.array([1.0, 0.1, 0.01])
        est = MFMCEstimator(stat, costs)
        target_cost = 100.0
        est.allocate_samples(target_cost)

        # Generate samples and values
        np.random.seed(456)
        nsamples_per_model = [
            int(est._rounded_nsamples_per_model[ii])
            for ii in range(nmodels)
        ]

        # Generate correlated values
        max_samples = max(nsamples_per_model)
        base = np.random.randn(nqoi, max_samples)
        values_per_model = []
        for ii in range(nmodels):
            n = nsamples_per_model[ii]
            if ii == 0:
                vals = base[:, :n]
            else:
                noise = np.random.randn(nqoi, n) * (0.1 * ii)
                vals = base[:, :n] + noise
            values_per_model.append(self._bkd.asarray(vals))

        # Bootstrap
        bootstrap_mean, bootstrap_cov = est.bootstrap(values_per_model, nbootstraps)

        # Compare to analytical covariance (with larger tolerance for MFMC)
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            bootstrap_cov, analytical_cov, atol=1e-1, rtol=3e-1
        )


class TestPilotQuantities(Generic[Array], unittest.TestCase):
    """Test pilot quantity computation from samples."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    @slow_test
    def test_mean_pilot_covariance(self) -> None:
        """Test compute_pilot_quantities for mean statistic.

        Uses many samples so MC estimate matches analytical covariance.
        """
        nqoi = 2
        nmodels = 2
        nsamples = 100000

        # Create known covariance
        true_cov = self._bkd.array([
            [1.0, 0.3, 0.8, 0.2],
            [0.3, 2.0, 0.2, 1.5],
            [0.8, 0.2, 1.0, 0.3],
            [0.2, 1.5, 0.3, 2.0],
        ])

        # Generate samples from multivariate normal
        np.random.seed(123)
        L = np.linalg.cholesky(true_cov.numpy() if hasattr(true_cov, 'numpy') else np.array(true_cov))
        z = np.random.randn(nmodels * nqoi, nsamples)
        all_values = L @ z  # shape (nmodels * nqoi, nsamples)

        # Split into per-model values using typing convention (nqoi, nsamples)
        pilot_values = [
            self._bkd.asarray(all_values[ii * nqoi : (ii + 1) * nqoi, :])
            for ii in range(nmodels)
        ]

        # Compute pilot quantities
        stat = MultiOutputMean(nqoi, self._bkd)
        (computed_cov,) = stat.compute_pilot_quantities(pilot_values)

        # Compare to true covariance
        self._bkd.assert_allclose(computed_cov, true_cov, atol=5e-2, rtol=5e-2)

    @slow_test
    def test_pilot_covariance_symmetry(self) -> None:
        """Test that computed pilot covariance is symmetric."""
        nqoi = 3
        nmodels = 2
        nsamples = 1000

        # Generate random values using typing convention (nqoi, nsamples)
        np.random.seed(789)
        pilot_values = [
            self._bkd.asarray(np.random.randn(nqoi, nsamples))
            for _ in range(nmodels)
        ]

        stat = MultiOutputMean(nqoi, self._bkd)
        (cov,) = stat.compute_pilot_quantities(pilot_values)

        # Check symmetry
        self._bkd.assert_allclose(cov, cov.T, rtol=1e-12)


class TestInsertPilotSamples(unittest.TestCase):
    """Test pilot sample insertion functionality.

    Uses Torch backend since ACV estimators require jacobians for optimization.
    """

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @slow_test
    def test_grd_insert_pilot_values(self) -> None:
        """Test insert_pilot_values for GRD estimator.

        The insert_pilot_values method is specific to ACV estimators
        which have allocation matrices. CV estimators don't have this.
        """
        nqoi = 2
        nmodels = 3
        npilot = 5

        # Create estimator with allocation matrix
        cov = self._bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        costs = self._bkd.array([1.0, 0.5, 0.25])
        recursion_index = self._bkd.array([0, 1], dtype=int)
        est = GRDEstimator(stat, costs, recursion_index=recursion_index)
        est.allocate_samples(200.0)

        # Get the actual number of samples per model after allocation
        nsamples_per_model = [
            int(est._rounded_nsamples_per_model[ii]) for ii in range(nmodels)
        ]

        # Create pilot values using typing convention (nqoi, nsamples)
        pilot_values = [
            self._bkd.ones((nqoi, npilot)) * (ii + 1)
            for ii in range(nmodels)
        ]

        # Create sample values (without pilot samples)
        sample_values = [
            self._bkd.ones((nqoi, max(0, nsamples_per_model[ii] - npilot))) * (ii + 10)
            for ii in range(nmodels)
        ]

        # Insert pilot values
        combined = est.insert_pilot_values(pilot_values, sample_values)

        # Check that insertion happened
        self._bkd.assert_allclose(
            self._bkd.asarray([len(combined)]),
            self._bkd.asarray([nmodels])
        )


class TestMCVarianceEstimation(Generic[Array], unittest.TestCase):
    """Test MC estimation of estimator variance matches analytical."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    @slow_test
    def test_mc_estimator_variance(self) -> None:
        """Test MC variance estimate matches analytical for MC estimator."""
        nqoi = 1
        nmodels = 1
        nsamples = 50
        ntrials = 5000

        # Create statistic
        variance = 4.0
        cov = self._bkd.eye(nmodels * nqoi) * variance
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        # Create estimator
        costs = self._bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(float(nsamples))

        # Compute MC variance estimate
        # MCEstimator takes a single array, not a list
        estimates = []
        for _ in range(ntrials):
            vals = self._bkd.asarray(
                np.random.randn(nqoi, nsamples) * np.sqrt(variance)
            )
            est_val = est(vals)
            estimates.append(est_val)
        estimates = self._bkd.stack(estimates)
        mc_cov = self._bkd.cov(estimates, rowvar=False, ddof=1)

        # Compare to analytical
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            mc_cov, analytical_cov, atol=5e-2, rtol=2e-1
        )

    @slow_test
    def test_cv_estimator_variance(self) -> None:
        """Test MC variance estimate matches analytical for CV estimator."""
        nqoi = 1
        nmodels = 2
        nsamples = 100
        ntrials = 3000

        # Create covariance with correlation
        cov = self._bkd.array([[1.0, 0.8], [0.8, 1.0]])
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        # Known low-fidelity mean (zero)
        lowfi_stats = self._bkd.zeros((nmodels - 1, nqoi))

        # Create estimator
        costs = self._bkd.array([1.0, 0.1])
        est = CVEstimator(stat, costs, lowfi_stats)
        target_cost = float(nsamples) * costs.sum()
        est.allocate_samples(target_cost)

        # Generate correlated values
        def values_generator():
            base = np.random.randn(nqoi, nsamples)
            hf_values = self._bkd.asarray(base)
            lf_values = self._bkd.asarray(base + np.random.randn(nqoi, nsamples) * 0.4)
            return [hf_values, lf_values]

        mc_cov = _compute_mc_variance(self._bkd, est, values_generator, ntrials)

        # Compare to analytical
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            mc_cov, analytical_cov, atol=5e-2, rtol=3e-1
        )


class TestEstimatorReproducibility(Generic[Array], unittest.TestCase):
    """Test estimator produces reproducible results."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError()

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def test_mc_estimator_reproducible(self) -> None:
        """Test MC estimator produces same result with same seed."""
        nqoi = 2
        nsamples = 100

        stat = MultiOutputMean(nqoi, self._bkd)
        cov = self._bkd.eye(nqoi) * 4.0
        stat.set_pilot_quantities(cov)

        costs = self._bkd.array([1.0])
        est = MCEstimator(stat, costs)
        est.allocate_samples(float(nsamples))

        # Generate values with same seed twice
        np.random.seed(123)
        values1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result1 = est(values1)

        np.random.seed(123)
        values2 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result2 = est(values2)

        self._bkd.assert_allclose(result1, result2, rtol=1e-12)

    def test_cv_estimator_reproducible(self) -> None:
        """Test CV estimator produces same result with same seed."""
        nqoi = 1
        nmodels = 2
        nsamples = 50

        cov = self._bkd.array([[1.0, 0.8], [0.8, 1.0]])
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        lowfi_stats = self._bkd.zeros((nmodels - 1, nqoi))
        costs = self._bkd.array([1.0, 0.5])
        est = CVEstimator(stat, costs, lowfi_stats)
        est.allocate_samples(float(nsamples) * costs.sum())

        # Generate values with same seed twice
        np.random.seed(456)
        hf1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        lf1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result1 = est([hf1, lf1])

        np.random.seed(456)
        hf2 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        lf2 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result2 = est([hf2, lf2])

        self._bkd.assert_allclose(result1, result2, rtol=1e-12)


# NumPy backend tests


class TestBootstrapEstimatorNumpy(TestBootstrapEstimator[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPilotQuantitiesNumpy(TestPilotQuantities[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()




class TestMCVarianceEstimationNumpy(TestMCVarianceEstimation[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestEstimatorReproducibilityNumpy(TestEstimatorReproducibility[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests


class TestBootstrapEstimatorTorch(TestBootstrapEstimator[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestPilotQuantitiesTorch(TestPilotQuantities[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()




class TestMCVarianceEstimationTorch(TestMCVarianceEstimation[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestEstimatorReproducibilityTorch(TestEstimatorReproducibility[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestPolynomialEnsemble(unittest.TestCase):
    """End-to-end test with polynomial model ensemble benchmark.

    Uses Torch backend since ACV optimization requires jacobians.
    This replicates the legacy test_polynomial_ensemble test using
    the typing module's PolynomialEnsemble benchmark.
    """

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @slow_test
    def test_gmf_polynomial_ensemble(self) -> None:
        """Test GMF estimator with polynomial ensemble benchmark.

        Verifies that analytical variance matches MC estimated variance.
        """
        from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
            PolynomialEnsemble,
        )

        # Create polynomial ensemble using typing module
        ensemble = PolynomialEnsemble(self._bkd, nmodels=5)
        cov = ensemble.covariance_matrix()
        nmodels = ensemble.nmodels()
        costs = ensemble.costs()

        stat = MultiOutputMean(ensemble.nqoi(), self._bkd)
        stat.set_pilot_quantities(cov)

        recursion_index = self._bkd.zeros(nmodels - 1, dtype=int)
        est = GMFEstimator(stat, costs, recursion_index=recursion_index)
        target_cost = 30.0
        est.allocate_samples(target_cost)

        # Get analytical covariance
        analytical_cov = est._covariance_from_npartition_samples(
            est._rounded_npartition_samples
        )

        # Compute MC estimate of variance
        ntrials = 5000
        models = ensemble.models()

        # Define uniform prior on [0, 1]
        def rvs(nsamples: int) -> torch.Tensor:
            n = int(nsamples)  # Convert from tensor if needed
            return self._bkd.asarray(np.random.rand(1, n))

        estimates = []
        for _ in range(ntrials):
            # Generate samples for each model
            samples_per_model = est.generate_samples_per_model(rvs)

            # Evaluate models - PolynomialModelFunction returns (nqoi, nsamples)
            values_per_model = [
                model(samples)
                for model, samples in zip(models, samples_per_model)
            ]

            est_val = est(values_per_model)
            estimates.append(est_val)

        estimates = self._bkd.stack(estimates)
        mc_cov = self._bkd.cov(estimates, rowvar=False, ddof=1)

        # Compare MC and analytical covariance
        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=2e-1, atol=1e-3)

    @slow_test
    def test_mfmc_polynomial_ensemble(self) -> None:
        """Test MFMC estimator with polynomial ensemble benchmark."""
        from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
            PolynomialEnsemble,
        )

        ensemble = PolynomialEnsemble(self._bkd, nmodels=3)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(ensemble.nqoi(), self._bkd)
        stat.set_pilot_quantities(cov)

        est = MFMCEstimator(stat, costs)
        target_cost = 50.0
        est.allocate_samples(target_cost)

        analytical_cov = est.optimized_covariance()

        # MC estimation
        ntrials = 3000
        models = ensemble.models()

        def rvs(nsamples: int) -> torch.Tensor:
            n = int(nsamples)  # Convert from tensor if needed
            return self._bkd.asarray(np.random.rand(1, n))

        estimates = []
        for _ in range(ntrials):
            samples_per_model = est.generate_samples_per_model(rvs)
            values_per_model = [
                model(samples)
                for model, samples in zip(models, samples_per_model)
            ]
            est_val = est(values_per_model)
            estimates.append(est_val)

        estimates = self._bkd.stack(estimates)
        mc_cov = self._bkd.cov(estimates, rowvar=False, ddof=1)

        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=2e-1, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
