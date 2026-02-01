"""Standalone tests for statest module.

These tests verify the correctness of the statest implementation
without comparing to legacy code. They test:
- Bootstrap variance estimation matches analytical variance
- Pilot quantity computation from samples
- End-to-end polynomial ensemble tests
- MC variance estimation comparing analytical to numerical
- Estimator variance across different estimator types, stat types, and nqoi
- BestEstimatorFactory selection procedure

These tests use the typing array convention: (nqoi, nsamples) for outputs.
"""

import unittest

import numpy as np
import torch
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test, slower_test  # noqa: F401

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
from pyapprox.typing.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
)
from pyapprox.typing.benchmarks.functions.multifidelity.multioutput_ensemble import (
    MultiOutputModelEnsemble,
)
from pyapprox.typing.statest.factory.best_estimator_factory import (
    BestEstimatorFactory,
)


def _get_stat(stat_type: str, nqoi: int, bkd):
    """Create statistic object by type."""
    if stat_type == "mean":
        return MultiOutputMean(nqoi, bkd)
    elif stat_type == "variance":
        return MultiOutputVariance(nqoi, bkd)
    elif stat_type == "mean_variance":
        return MultiOutputMeanAndVariance(nqoi, bkd)
    else:
        raise ValueError(f"Unknown stat_type: {stat_type}")


def _get_estimator(est_type: str, stat, costs, bkd, **kwargs):
    """Create estimator by type."""
    if est_type == "mc":
        return MCEstimator(stat, costs)
    elif est_type == "cv":
        return CVEstimator(stat, costs, kwargs.get("lowfi_stats"))
    elif est_type == "mfmc":
        return MFMCEstimator(stat, costs)
    elif est_type == "mlmc":
        return MLMCEstimator(stat, costs)
    elif est_type == "gmf":
        return GMFEstimator(stat, costs, recursion_index=kwargs.get("recursion_index"))
    elif est_type == "grd":
        return GRDEstimator(stat, costs, recursion_index=kwargs.get("recursion_index"))
    elif est_type == "gis":
        return GISEstimator(stat, costs, recursion_index=kwargs.get("recursion_index"))
    else:
        raise ValueError(f"Unknown est_type: {est_type}")


def _setup_pilot_quantities(stat_type: str, nmodels: int, nqoi: int, bkd):
    """Create pilot quantities for a given stat type."""
    # Create a positive definite covariance matrix
    np.random.seed(12345)
    A = np.random.randn(nmodels * nqoi, nmodels * nqoi)
    cov = bkd.asarray(A @ A.T / (nmodels * nqoi) + np.eye(nmodels * nqoi) * 0.1)

    pilot_args = [cov]

    if "variance" in stat_type:
        # W matrix for variance estimator covariance
        W_size = nmodels * nqoi**2
        W = bkd.eye(W_size) * 0.5
        pilot_args.append(W)

    if stat_type == "mean_variance":
        # B matrix for mean-variance cross-covariance
        B = bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
        pilot_args.append(B)

    return pilot_args


def _compute_mc_estimator_variance(
    bkd, ensemble, est, ntrials: int
) -> torch.Tensor:
    """Compute MC estimate of estimator variance using polynomial ensemble."""
    models = ensemble.models()
    nmodels = ensemble.nmodels()

    def rvs(nsamples):
        n = int(nsamples)
        return bkd.asarray(np.random.rand(1, n))

    estimates = []
    for _ in range(ntrials):
        samples_per_model = est.generate_samples_per_model(rvs)
        values_per_model = [
            model(samples)
            for model, samples in zip(models, samples_per_model)
        ]
        # MCEstimator takes a single array, not a list
        if nmodels == 1:
            est_val = est(values_per_model[0])
        else:
            est_val = est(values_per_model)
        estimates.append(est_val)

    estimates = bkd.stack(estimates)
    return bkd.cov(estimates, rowvar=False, ddof=1)


class TestBootstrapEstimator(ParametrizedTestCase):
    """Test bootstrap variance estimation matches analytical variance."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "est_type,target_cost",
        [
            ("mc", 1000),
            ("cv", 500),
        ],
        ids=["mc", "cv"],
    )
    @slow_test
    def test_bootstrap_variance(self, est_type: str, target_cost: float) -> None:
        """Test bootstrap variance matches analytical for MC and CV estimators."""
        nqoi = 2
        nmodels = 3 if est_type != "mc" else 1
        nbootstraps = 500

        # Create covariance with correlations
        np.random.seed(123)
        A = np.random.randn(nmodels * nqoi, nmodels * nqoi)
        cov = self._bkd.asarray(A @ A.T / (nmodels * nqoi) + np.eye(nmodels * nqoi))

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        costs = self._bkd.array([1.0, 0.1, 0.01][:nmodels])

        if est_type == "cv":
            lowfi_stats = self._bkd.zeros((nmodels - 1, nqoi))
            est = CVEstimator(stat, costs, lowfi_stats)
        else:
            est = MCEstimator(stat, costs)

        est.allocate_samples(target_cost)

        # Generate values
        np.random.seed(456)
        nsamples_per_model = [
            int(est._rounded_nsamples_per_model[ii]) for ii in range(nmodels)
        ]
        max_samples = max(nsamples_per_model)

        # Generate correlated values
        L = np.linalg.cholesky(cov.numpy())
        base = L @ np.random.randn(nmodels * nqoi, max_samples)

        values_per_model = []
        for ii in range(nmodels):
            n = nsamples_per_model[ii]
            vals = base[ii * nqoi : (ii + 1) * nqoi, :n]
            values_per_model.append(self._bkd.asarray(vals))

        # Bootstrap
        bootstrap_mean, bootstrap_cov = est.bootstrap(values_per_model, nbootstraps)

        # Compare to analytical covariance
        analytical_cov = est.optimized_covariance()
        self._bkd.assert_allclose(
            bootstrap_cov, analytical_cov, atol=1e-1, rtol=3e-1
        )

    @parametrize(
        "est_type,target_cost",
        [
            ("mfmc", 5000),
            ("gmf", 5000),
            ("grd", 5000),
            ("gis", 5000),
        ],
        ids=["mfmc", "gmf", "grd", "gis"],
    )
    @slow_test
    def test_bootstrap_variance_acv(self, est_type: str, target_cost: float) -> None:
        """Test bootstrap variance for ACV estimators requiring hierarchical cov."""
        nqoi = 1
        nmodels = 3
        nbootstraps = 500

        # Use PolynomialEnsemble for proper hierarchical covariance
        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        costs = self._bkd.array([1.0, 0.1, 0.01])
        recursion_index = self._bkd.array([0, 1], dtype=int)

        est = _get_estimator(
            est_type, stat, costs, self._bkd, recursion_index=recursion_index
        )
        est.allocate_samples(target_cost)

        # Generate samples and values
        np.random.seed(789)
        nsamples_per_model = [
            int(est._rounded_nsamples_per_model[ii]) for ii in range(nmodels)
        ]
        max_samples = max(nsamples_per_model)

        L = np.linalg.cholesky(cov.numpy())
        base = L @ np.random.randn(nmodels * nqoi, max_samples)

        values_per_model = []
        for ii in range(nmodels):
            n = nsamples_per_model[ii]
            vals = base[ii * nqoi : (ii + 1) * nqoi, :n]
            values_per_model.append(self._bkd.asarray(vals))

        bootstrap_mean, bootstrap_cov = est.bootstrap(values_per_model, nbootstraps)
        analytical_cov = est.optimized_covariance()

        self._bkd.assert_allclose(
            bootstrap_cov, analytical_cov, atol=1e-1, rtol=5e-1
        )


class TestEstimatorVariance(ParametrizedTestCase):
    """Test estimator variance computation across different configurations.

    Verifies that MC estimated variance matches analytical variance formula.
    """

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "est_type,stat_type,nqoi",
        [
            # MC tests - nmodels=1, uses single array input
            ("mc", "mean", 1),
            ("mc", "variance", 1),
            ("mc", "mean_variance", 1),
            # CV tests - use PolynomialEnsemble (nqoi=1)
            ("cv", "mean", 1),
            # MFMC/MLMC - use PolynomialEnsemble (nqoi=1)
            ("mfmc", "mean", 1),
            ("mlmc", "mean", 1),
        ],
        ids=[
            "mc_mean_1qoi",
            "mc_variance_1qoi",
            "mc_mean_variance_1qoi",
            "cv_mean_1qoi",
            "mfmc_mean_1qoi",
            "mlmc_mean_1qoi",
        ],
    )
    @slower_test
    def test_variance_matches_mc(
        self, est_type: str, stat_type: str, nqoi: int
    ) -> None:
        """Test analytical variance matches MC estimate with single QoI."""
        nmodels = 1 if est_type == "mc" else 3
        ntrials = 3000
        target_cost = 100.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = _get_stat(stat_type, nqoi, self._bkd)

        # Set up pilot quantities
        pilot_args = [cov]
        if "variance" in stat_type:
            W_size = nmodels * nqoi**2
            W = self._bkd.eye(W_size) * 0.1
            pilot_args.append(W)
        if stat_type == "mean_variance":
            B = self._bkd.zeros((nmodels * nqoi, nmodels * nqoi**2))
            pilot_args.append(B)

        stat.set_pilot_quantities(*pilot_args)

        # Create estimator
        kwargs = {}
        if est_type == "cv":
            kwargs["lowfi_stats"] = self._bkd.zeros((nmodels - 1, stat.nstats()))

        est = _get_estimator(est_type, stat, costs, self._bkd, **kwargs)
        est.allocate_samples(target_cost)

        # Compute MC variance
        mc_cov = _compute_mc_estimator_variance(self._bkd, ensemble, est, ntrials)
        analytical_cov = est.optimized_covariance()

        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=3e-1, atol=5e-2)

    @parametrize(
        "est_type,model_idx,qoi_idx",
        [
            # CV with 2 QoI using MultiOutputModelEnsemble
            ("cv", [0, 1, 2], [0, 1]),
            # MFMC with 2 QoI
            ("mfmc", [0, 1, 2], [0, 1]),
        ],
        ids=[
            "cv_mean_2qoi",
            "mfmc_mean_2qoi",
        ],
    )
    @slower_test
    def test_variance_matches_mc_multioutput(
        self, est_type: str, model_idx: list, qoi_idx: list
    ) -> None:
        """Test analytical variance matches MC estimate with multiple QoI."""
        ntrials = 2000
        target_cost = 50.0

        ensemble = MultiOutputModelEnsemble(self._bkd)
        nmodels = len(model_idx)
        nqoi = len(qoi_idx)

        # Get subproblem covariance and costs
        cov = ensemble.covariance_subproblem(model_idx, qoi_idx)
        costs = ensemble.costs_subproblem(model_idx)
        models = ensemble.models_subproblem(model_idx, qoi_idx)

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        # Create estimator
        kwargs = {}
        if est_type == "cv":
            kwargs["lowfi_stats"] = self._bkd.zeros((nmodels - 1, stat.nstats()))

        est = _get_estimator(est_type, stat, costs, self._bkd, **kwargs)
        est.allocate_samples(target_cost)

        # Compute MC variance using subproblem models
        def rvs(nsamples):
            n = int(nsamples)
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
        analytical_cov = est.optimized_covariance()

        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=3e-1, atol=5e-2)

    @parametrize(
        "est_type,stat_type,recursion_index",
        [
            ("gmf", "mean", [0, 0]),
            ("gmf", "mean", [0, 1]),
            ("grd", "mean", [0, 0]),
            ("grd", "mean", [0, 1]),
            ("gis", "mean", [0, 1]),
        ],
        ids=[
            "gmf_mean_00",
            "gmf_mean_01",
            "grd_mean_00",
            "grd_mean_01",
            "gis_mean_01",
        ],
    )
    @slow_test
    def test_variance_matches_mc_acv(
        self, est_type: str, stat_type: str, recursion_index: list
    ) -> None:
        """Test analytical variance matches MC estimate for ACV estimators."""
        nmodels = 3
        nqoi = 1
        ntrials = 2000
        target_cost = 50.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = _get_stat(stat_type, nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        rec_idx = self._bkd.array(recursion_index, dtype=int)
        est = _get_estimator(est_type, stat, costs, self._bkd, recursion_index=rec_idx)
        est.allocate_samples(target_cost)

        mc_cov = _compute_mc_estimator_variance(self._bkd, ensemble, est, ntrials)
        analytical_cov = est.optimized_covariance()

        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=3e-1, atol=5e-2)


class TestPilotQuantities(ParametrizedTestCase):
    """Test pilot quantity computation from samples."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "nqoi,nmodels",
        [
            (1, 2),
            (2, 2),
            (1, 3),
            (2, 3),
        ],
        ids=["1qoi_2mod", "2qoi_2mod", "1qoi_3mod", "2qoi_3mod"],
    )
    @slow_test
    def test_pilot_covariance_from_samples(self, nqoi: int, nmodels: int) -> None:
        """Test compute_pilot_quantities matches known covariance."""
        nsamples = 50000

        # Create known covariance
        np.random.seed(123)
        size = nmodels * nqoi
        A = np.random.randn(size, size)
        true_cov = self._bkd.asarray(A @ A.T / size + np.eye(size) * 0.5)

        # Generate samples from multivariate normal
        L = np.linalg.cholesky(true_cov.numpy())
        z = np.random.randn(size, nsamples)
        all_values = L @ z

        # Split into per-model values (nqoi, nsamples)
        pilot_values = [
            self._bkd.asarray(all_values[ii * nqoi : (ii + 1) * nqoi, :])
            for ii in range(nmodels)
        ]

        stat = MultiOutputMean(nqoi, self._bkd)
        (computed_cov,) = stat.compute_pilot_quantities(pilot_values)

        self._bkd.assert_allclose(computed_cov, true_cov, atol=5e-2, rtol=5e-2)

    def test_pilot_covariance_symmetry(self) -> None:
        """Test that computed pilot covariance is symmetric."""
        nqoi = 3
        nmodels = 2
        nsamples = 1000

        np.random.seed(789)
        pilot_values = [
            self._bkd.asarray(np.random.randn(nqoi, nsamples))
            for _ in range(nmodels)
        ]

        stat = MultiOutputMean(nqoi, self._bkd)
        (cov,) = stat.compute_pilot_quantities(pilot_values)

        self._bkd.assert_allclose(cov, cov.T, rtol=1e-12)


class TestPolynomialEnsemble(ParametrizedTestCase):
    """End-to-end tests with polynomial model ensemble benchmark."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "est_type,nmodels,recursion_index",
        [
            ("mfmc", 3, None),
            ("mfmc", 5, None),
            ("mlmc", 3, None),
            ("gmf", 3, [0, 0]),
            ("gmf", 5, [0, 0, 0, 0]),
            ("grd", 3, [0, 1]),
        ],
        ids=["mfmc_3", "mfmc_5", "mlmc_3", "gmf_3", "gmf_5", "grd_3"],
    )
    @slow_test
    def test_polynomial_ensemble(
        self, est_type: str, nmodels: int, recursion_index: list
    ) -> None:
        """Test estimator with polynomial ensemble benchmark."""
        ntrials = 3000
        target_cost = 30.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(ensemble.nqoi(), self._bkd)
        stat.set_pilot_quantities(cov)

        kwargs = {}
        if recursion_index is not None:
            kwargs["recursion_index"] = self._bkd.array(recursion_index, dtype=int)

        est = _get_estimator(est_type, stat, costs, self._bkd, **kwargs)
        est.allocate_samples(target_cost)

        analytical_cov = est.optimized_covariance()
        mc_cov = _compute_mc_estimator_variance(self._bkd, ensemble, est, ntrials)

        self._bkd.assert_allclose(mc_cov, analytical_cov, rtol=2e-1, atol=1e-2)


class TestInsertPilotSamples(ParametrizedTestCase):
    """Test pilot sample insertion functionality."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @parametrize(
        "est_type,recursion_index",
        [
            ("grd", [0, 1]),
            ("gmf", [0, 0]),
            ("gis", [0, 1]),
        ],
        ids=["grd", "gmf", "gis"],
    )
    @slow_test
    def test_insert_pilot_values(self, est_type: str, recursion_index: list) -> None:
        """Test insert_pilot_values for ACV estimators."""
        nqoi = 2
        nmodels = 3
        npilot = 5

        cov = self._bkd.eye(nmodels * nqoi)
        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)
        costs = self._bkd.array([1.0, 0.5, 0.25])
        rec_idx = self._bkd.array(recursion_index, dtype=int)

        est = _get_estimator(est_type, stat, costs, self._bkd, recursion_index=rec_idx)
        est.allocate_samples(200.0)

        nsamples_per_model = [
            int(est._rounded_nsamples_per_model[ii]) for ii in range(nmodels)
        ]

        pilot_values = [
            self._bkd.ones((nqoi, npilot)) * (ii + 1)
            for ii in range(nmodels)
        ]
        sample_values = [
            self._bkd.ones((nqoi, max(0, nsamples_per_model[ii] - npilot))) * (ii + 10)
            for ii in range(nmodels)
        ]

        combined = est.insert_pilot_values(pilot_values, sample_values)

        self._bkd.assert_allclose(
            self._bkd.asarray([len(combined)]),
            self._bkd.asarray([nmodels])
        )


class TestEstimatorReproducibility(unittest.TestCase):
    """Test estimator produces reproducible results."""

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

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

        np.random.seed(456)
        hf1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        lf1 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result1 = est([hf1, lf1])

        np.random.seed(456)
        hf2 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        lf2 = self._bkd.asarray(np.random.randn(nqoi, nsamples))
        result2 = est([hf2, lf2])

        self._bkd.assert_allclose(result1, result2, rtol=1e-12)


class TestBestEstimatorFactory(ParametrizedTestCase):
    """Test BestEstimatorFactory selects optimal estimator configuration.

    Focus: Test selection procedure correctness, not individual estimator
    correctness (already tested elsewhere).
    """

    def setUp(self) -> None:
        np.random.seed(42)
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()

    @slow_test
    def test_best_estimator_selection(self) -> None:
        """Test that BestEstimatorFactory selects the best configuration."""
        bkd = self._bkd

        # Use polynomial ensemble for clean analytical covariance
        nmodels = 3
        ensemble = PolynomialEnsemble(bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(ensemble.nqoi(), bkd)
        stat.set_pilot_quantities(cov)

        target_cost = 50.0

        # Create BestEstimatorFactory with save_candidates=True
        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf", "gis", "grd"],
            max_nmodels=nmodels,
            save_candidates=True,
            verbosity=0,
        )
        factory.allocate_samples(target_cost)

        # Get all successful candidate results
        candidates = factory.candidate_results()
        successful = [c for c in candidates if c.success]
        self.assertGreater(len(successful), 0)

        # Verify best has minimum objective value
        best_obj = factory.best_objective_value()
        all_objectives = [c.objective_value for c in successful]
        self._bkd.assert_allclose(
            self._bkd.asarray([best_obj]),
            self._bkd.asarray([min(all_objectives)]),
            rtol=1e-10,
        )

        # Verify best estimator is returned and is valid
        best_est = factory.best_estimator()
        self.assertIsNotNone(best_est)
        self.assertIsNotNone(factory.best_type())
        self.assertIsNotNone(factory.best_models())

    def test_best_estimator_requires_allocate(self) -> None:
        """Test that best_estimator raises before allocate_samples."""
        bkd = self._bkd
        ensemble = PolynomialEnsemble(bkd, nmodels=3)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(ensemble.nqoi(), bkd)
        stat.set_pilot_quantities(cov)

        factory = BestEstimatorFactory(stat, costs, bkd)

        with self.assertRaises(ValueError):
            factory.best_estimator()

    @slow_test
    def test_best_estimator_callable(self) -> None:
        """Test that BestEstimatorFactory can be called like an estimator."""
        bkd = self._bkd

        nmodels = 3
        ensemble = PolynomialEnsemble(bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()
        models = ensemble.models()

        stat = MultiOutputMean(ensemble.nqoi(), bkd)
        stat.set_pilot_quantities(cov)

        target_cost = 50.0

        factory = BestEstimatorFactory(
            stat,
            costs,
            bkd,
            estimator_types=["gmf"],
            max_nmodels=nmodels,
            verbosity=0,
        )
        factory.allocate_samples(target_cost)

        # Generate samples and compute estimate via factory
        def rvs(n: int):
            return bkd.asarray(np.random.rand(1, int(n)))

        best_est = factory.best_estimator()
        samples_per_model = best_est.generate_samples_per_model(rvs)

        # Only use models in best_models
        best_models = factory.best_models()
        values_per_model = [
            models[idx](samples_per_model[ii])
            for ii, idx in enumerate(best_models)
        ]

        # Call via factory
        result = factory(values_per_model)
        self.assertEqual(result.shape[0], stat.nstats())


if __name__ == "__main__":
    unittest.main()
