"""Integration tests for ACV estimators.

These tests replicate the legacy integration tests from
pyapprox/multifidelity/tests/test_acv.py to ensure the typing
implementation produces identical results.

# TODO: Delete after refactor complete
"""

import unittest

import numpy as np
import torch

from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    allocate_with_allocator,
)

# Legacy imports
from pyapprox.util.backends.torch import TorchMixin as LegacyTorchBackend
from pyapprox.multifidelity.factory import (
    get_estimator as legacy_get_estimator,
    multioutput_stats as legacy_multioutput_stats,
)
from pyapprox.multifidelity.tests.test_stats import (
    _setup_multioutput_model_subproblem,
)
from pyapprox.multifidelity.stats import (
    _nqoisq_nqoisq_subproblem,
    _nqoi_nqoisq_subproblem,
)

# Typing imports
from pyapprox.typing.statest.statistics import (
    MultiOutputMean as TypingMultiOutputMean,
    MultiOutputVariance as TypingMultiOutputVariance,
    MultiOutputMeanAndVariance as TypingMultiOutputMeanAndVariance,
)
from pyapprox.typing.statest.mc_estimator import MCEstimator as TypingMCEstimator
from pyapprox.typing.statest.cv_estimator import CVEstimator as TypingCVEstimator
from pyapprox.typing.statest.acv.variants import (
    GMFEstimator as TypingGMFEstimator,
    GISEstimator as TypingGISEstimator,
    GRDEstimator as TypingGRDEstimator,
    MFMCEstimator as TypingMFMCEstimator,
    MLMCEstimator as TypingMLMCEstimator,
)


# Typing stat factory
typing_multioutput_stats = {
    "mean": lambda nqoi, bkd: TypingMultiOutputMean(nqoi, bkd),
    "variance": lambda nqoi, bkd: TypingMultiOutputVariance(nqoi, bkd),
    "mean_variance": lambda nqoi, bkd: TypingMultiOutputMeanAndVariance(nqoi, bkd),
}


def _get_typing_estimator(est_type, stat, costs, bkd, **kwargs):
    """Factory function to create typing estimators."""
    if est_type == "mc":
        return TypingMCEstimator(stat, costs)
    elif est_type == "cv":
        return TypingCVEstimator(stat, costs, kwargs.get("lowfi_stats"))
    elif est_type == "gmf":
        return TypingGMFEstimator(
            stat, costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    elif est_type == "gis":
        return TypingGISEstimator(
            stat, costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    elif est_type == "grd":
        return TypingGRDEstimator(
            stat, costs,
            recursion_index=kwargs.get("recursion_index"),
            tree_depth=kwargs.get("tree_depth"),
        )
    elif est_type == "mfmc":
        return TypingMFMCEstimator(stat, costs)
    elif est_type == "mlmc":
        return TypingMLMCEstimator(stat, costs)
    else:
        raise ValueError(f"Unknown estimator type: {est_type}")


class TestLegacyComparisonAllocationMatrices(unittest.TestCase):
    """Compare allocation matrices between legacy and typing.

    Uses torch backend since GMF/GIS/GRD require jacobians.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(1)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    @slow_test
    def test_grd_allocation_matrices(self) -> None:
        """Compare GRD allocation matrices."""
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(
                model_idx, qoi_idx, self._legacy_bkd
            )
        )

        # Test case 1: recursion_index=[0, 0]
        legacy_stat = legacy_multioutput_stats["mean"](
            len(qoi_idx), backend=self._legacy_bkd
        )
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = legacy_get_estimator(
            "grd", legacy_stat, costs, recursion_index=[0, 0]
        )

        typing_stat = typing_multioutput_stats["mean"](
            len(qoi_idx), self._typing_bkd
        )
        typing_stat.set_pilot_quantities(cov)
        typing_est = _get_typing_estimator(
            "grd", typing_stat, costs, self._typing_bkd,
            recursion_index=torch.tensor([0, 0])
        )

        np.testing.assert_allclose(
            typing_est._allocation_mat.numpy(),
            legacy_est._allocation_mat.numpy(),
            rtol=1e-12
        )

    @slow_test
    def test_gmf_allocation_matrices(self) -> None:
        """Compare GMF allocation matrices."""
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(
                model_idx, qoi_idx, self._legacy_bkd
            )
        )

        legacy_stat = legacy_multioutput_stats["mean"](
            len(qoi_idx), backend=self._legacy_bkd
        )
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = legacy_get_estimator(
            "gmf", legacy_stat, costs, recursion_index=[0, 0]
        )

        typing_stat = typing_multioutput_stats["mean"](
            len(qoi_idx), self._typing_bkd
        )
        typing_stat.set_pilot_quantities(cov)
        typing_est = _get_typing_estimator(
            "gmf", typing_stat, costs, self._typing_bkd,
            recursion_index=torch.tensor([0, 0])
        )

        np.testing.assert_allclose(
            typing_est._allocation_mat.numpy(),
            legacy_est._allocation_mat.numpy(),
            rtol=1e-12
        )

    @slow_test
    def test_gis_allocation_matrices(self) -> None:
        """Compare GIS allocation matrices."""
        model_idx = [0, 1, 2]
        qoi_idx = [0]
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(
                model_idx, qoi_idx, self._legacy_bkd
            )
        )

        legacy_stat = legacy_multioutput_stats["mean"](
            len(qoi_idx), backend=self._legacy_bkd
        )
        legacy_stat.set_pilot_quantities(cov)
        legacy_est = legacy_get_estimator(
            "gis", legacy_stat, costs, recursion_index=[0, 1]
        )

        typing_stat = typing_multioutput_stats["mean"](
            len(qoi_idx), self._typing_bkd
        )
        typing_stat.set_pilot_quantities(cov)
        typing_est = _get_typing_estimator(
            "gis", typing_stat, costs, self._typing_bkd,
            recursion_index=torch.tensor([0, 1])
        )

        np.testing.assert_allclose(
            typing_est._allocation_mat.numpy(),
            legacy_est._allocation_mat.numpy(),
            rtol=1e-12
        )


class TestLegacyComparisonSampleAllocation(unittest.TestCase):
    """Compare sample allocation between legacy and typing.

    Uses torch backend since ACV estimators require jacobians for optimization.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(1)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    def _check_allocation(
        self,
        est_type: str,
        model_idx: list,
        qoi_idx: list,
        stat_type: str = "mean",
        recursion_index: list = None,
        target_cost: float = 50.0,
        rtol: float = 1e-6,
    ) -> None:
        """Compare sample allocation between legacy and typing."""
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(
                model_idx, qoi_idx, self._legacy_bkd
            )
        )
        # Use simpler costs
        costs = torch.tensor([2.0, 1.5, 1.0])[: len(model_idx)]

        nqoi = len(qoi_idx)

        # Build pilot_args based on stat_type
        pilot_args = [cov]
        if "variance" in stat_type:
            W = benchmark.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, benchmark.nmodels(), benchmark.nqoi(),
                model_idx, qoi_idx, self._legacy_bkd
            )
            pilot_args.append(W)
        if stat_type == "mean_variance":
            B = benchmark.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, benchmark.nmodels(), benchmark.nqoi(),
                model_idx, qoi_idx, self._legacy_bkd
            )
            pilot_args.append(B)

        # Setup kwargs
        legacy_kwargs = {}
        typing_kwargs = {}
        if recursion_index is not None:
            legacy_kwargs["recursion_index"] = recursion_index
            typing_kwargs["recursion_index"] = torch.tensor(recursion_index)

        # Legacy
        legacy_stat = legacy_multioutput_stats[stat_type](
            nqoi, backend=self._legacy_bkd
        )
        legacy_stat.set_pilot_quantities(*pilot_args)
        legacy_est = legacy_get_estimator(
            est_type, legacy_stat, costs, **legacy_kwargs
        )
        if hasattr(legacy_est, "get_default_optimizer"):
            optimizer = legacy_est.get_default_optimizer()
            optimizer.set_verbosity(0)
            optimizer._optimizer1._opts["maxiter"] = 500
            optimizer._optimizer2._opts["method"] = "slsqp"
            legacy_est.set_optimizer(optimizer)
        legacy_est.allocate_samples(target_cost)

        # Typing
        typing_stat = typing_multioutput_stats[stat_type](nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(*pilot_args)
        typing_est = _get_typing_estimator(
            est_type, typing_stat, costs, self._typing_bkd, **typing_kwargs
        )
        allocate_with_allocator(typing_est, target_cost)

        # Compare results
        np.testing.assert_allclose(
            typing_est._rounded_npartition_samples.numpy(),
            legacy_est._rounded_npartition_samples.numpy(),
            rtol=rtol
        )
        np.testing.assert_allclose(
            typing_est.optimized_covariance().numpy(),
            legacy_est.optimized_covariance().numpy(),
            rtol=rtol
        )

    # Mean statistic tests (single QoI)
    @slow_test
    def test_mfmc_allocation_mean(self) -> None:
        """Compare MFMC sample allocation with mean statistic."""
        self._check_allocation("mfmc", [0, 1, 2], [0], stat_type="mean")

    @slow_test
    def test_mlmc_allocation_mean(self) -> None:
        """Compare MLMC sample allocation with mean statistic."""
        self._check_allocation("mlmc", [0, 1, 2], [0], stat_type="mean")

    @slow_test
    def test_gmf_allocation_mean(self) -> None:
        """Compare GMF sample allocation with mean statistic."""
        self._check_allocation(
            "gmf", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 0]
        )

    @slow_test
    def test_grd_allocation_mean(self) -> None:
        """Compare GRD sample allocation with mean statistic."""
        self._check_allocation(
            "grd", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 0]
        )

    @slow_test
    def test_gis_allocation_mean(self) -> None:
        """Compare GIS sample allocation with mean statistic."""
        self._check_allocation(
            "gis", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 1]
        )

    # Variance statistic tests (single QoI)
    @slow_test
    def test_mfmc_allocation_variance(self) -> None:
        """Compare MFMC sample allocation with variance statistic."""
        self._check_allocation("mfmc", [0, 1, 2], [0], stat_type="variance")

    @slow_test
    def test_mlmc_allocation_variance(self) -> None:
        """Compare MLMC sample allocation with variance statistic."""
        self._check_allocation("mlmc", [0, 1, 2], [0], stat_type="variance")

    @slow_test
    def test_gmf_allocation_variance(self) -> None:
        """Compare GMF sample allocation with variance statistic."""
        self._check_allocation(
            "gmf", [0, 1], [0], stat_type="variance", recursion_index=[0]
        )

    @slow_test
    def test_grd_allocation_variance(self) -> None:
        """Compare GRD sample allocation with variance statistic."""
        self._check_allocation(
            "grd", [0, 1, 2], [0, 1], stat_type="variance", recursion_index=[0, 1]
        )

    # Mean-variance statistic tests
    @slow_test
    def test_gmf_allocation_mean_variance(self) -> None:
        """Compare GMF sample allocation with mean_variance statistic."""
        self._check_allocation(
            "gmf", [0, 1], [0], stat_type="mean_variance", recursion_index=[0]
        )

    @slow_test
    def test_mlmc_allocation_mean_variance(self) -> None:
        """Compare MLMC sample allocation with mean_variance statistic."""
        self._check_allocation(
            "mlmc", [0, 1, 2], [0], stat_type="mean_variance"
        )

    # Multi-QoI tests
    @slow_test
    def test_mfmc_allocation_mean_multi_qoi(self) -> None:
        """Compare MFMC sample allocation with mean statistic and 2 QoIs."""
        self._check_allocation("mfmc", [0, 1, 2], [0, 1], stat_type="mean")

    @slow_test
    def test_grd_allocation_mean_multi_qoi(self) -> None:
        """Compare GRD sample allocation with mean statistic and 2 QoIs."""
        self._check_allocation(
            "grd", [0, 1, 2], [0, 1], stat_type="mean", recursion_index=[0, 0]
        )

    @slow_test
    def test_grd_allocation_variance_multi_qoi(self) -> None:
        """Compare GRD sample allocation with variance statistic and 2 QoIs."""
        self._check_allocation(
            "grd", [0, 1, 2], [0, 1], stat_type="variance",
            recursion_index=[0, 1], target_cost=200.0
        )


class TestLegacyComparisonEstimatorVariance(unittest.TestCase):
    """Compare estimator variance computation between legacy and typing.

    These tests verify that the optimized variance formulas match between
    legacy and typing implementations.
    """

    # TODO: Delete after refactor complete

    def setUp(self) -> None:
        np.random.seed(1)
        torch.set_default_dtype(torch.float64)
        self._legacy_bkd = LegacyTorchBackend
        self._typing_bkd = TorchBkd()

    def _check_variance_formula(
        self,
        est_type: str,
        model_idx: list,
        qoi_idx: list,
        stat_type: str = "mean",
        recursion_index: list = None,
        target_cost: float = 50.0,
    ) -> None:
        """Compare variance formula between legacy and typing after allocation."""
        funs, cov, costs, benchmark, means = (
            _setup_multioutput_model_subproblem(
                model_idx, qoi_idx, self._legacy_bkd
            )
        )
        costs = torch.tensor([2.0, 1.5, 1.0])[: len(model_idx)]
        nqoi = len(qoi_idx)

        # Build pilot_args based on stat_type
        pilot_args = [cov]
        if "variance" in stat_type:
            W = benchmark.covariance_of_centered_values_kronker_product()
            W = _nqoisq_nqoisq_subproblem(
                W, benchmark.nmodels(), benchmark.nqoi(),
                model_idx, qoi_idx, self._legacy_bkd
            )
            pilot_args.append(W)
        if stat_type == "mean_variance":
            B = benchmark.covariance_of_mean_and_variance_estimators()
            B = _nqoi_nqoisq_subproblem(
                B, benchmark.nmodels(), benchmark.nqoi(),
                model_idx, qoi_idx, self._legacy_bkd
            )
            pilot_args.append(B)

        legacy_kwargs = {}
        typing_kwargs = {}
        if recursion_index is not None:
            legacy_kwargs["recursion_index"] = recursion_index
            typing_kwargs["recursion_index"] = torch.tensor(recursion_index)

        # Legacy
        legacy_stat = legacy_multioutput_stats[stat_type](
            nqoi, backend=self._legacy_bkd
        )
        legacy_stat.set_pilot_quantities(*pilot_args)
        legacy_est = legacy_get_estimator(
            est_type, legacy_stat, costs, **legacy_kwargs
        )
        if hasattr(legacy_est, "get_default_optimizer"):
            optimizer = legacy_est.get_default_optimizer()
            optimizer.set_verbosity(0)
            optimizer._optimizer1._opts["maxiter"] = 500
            optimizer._optimizer2._opts["method"] = "slsqp"
            legacy_est.set_optimizer(optimizer)
        legacy_est.allocate_samples(target_cost)

        # Typing
        typing_stat = typing_multioutput_stats[stat_type](nqoi, self._typing_bkd)
        typing_stat.set_pilot_quantities(*pilot_args)
        typing_est = _get_typing_estimator(
            est_type, typing_stat, costs, self._typing_bkd, **typing_kwargs
        )
        allocate_with_allocator(typing_est, target_cost)

        # Get discrepancy covariances
        if est_type not in ["mc"]:
            legacy_CF, legacy_cf = legacy_est._get_discrepancy_covariances(
                legacy_est._rounded_npartition_samples
            )
            typing_CF, typing_cf = typing_est._get_discrepancy_covariances(
                typing_est._rounded_npartition_samples
            )

            np.testing.assert_allclose(
                typing_CF.numpy(),
                legacy_CF.numpy(),
                rtol=1e-10
            )
            np.testing.assert_allclose(
                typing_cf.numpy(),
                legacy_cf.numpy(),
                rtol=1e-10
            )

    # Mean statistic tests (single QoI)
    @slow_test
    def test_mfmc_variance_mean(self) -> None:
        """Compare MFMC variance formula with mean statistic."""
        self._check_variance_formula("mfmc", [0, 1, 2], [0], stat_type="mean")

    @slow_test
    def test_mlmc_variance_mean(self) -> None:
        """Compare MLMC variance formula with mean statistic."""
        self._check_variance_formula("mlmc", [0, 1, 2], [0], stat_type="mean")

    @slow_test
    def test_gmf_variance_mean(self) -> None:
        """Compare GMF variance formula with mean statistic."""
        self._check_variance_formula(
            "gmf", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 0]
        )

    @slow_test
    def test_grd_variance_mean(self) -> None:
        """Compare GRD variance formula with mean statistic."""
        self._check_variance_formula(
            "grd", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 0]
        )

    @slow_test
    def test_gis_variance_mean(self) -> None:
        """Compare GIS variance formula with mean statistic."""
        self._check_variance_formula(
            "gis", [0, 1, 2], [0], stat_type="mean", recursion_index=[0, 1]
        )

    # Variance statistic tests
    @slow_test
    def test_mfmc_variance_variance_stat(self) -> None:
        """Compare MFMC variance formula with variance statistic."""
        self._check_variance_formula("mfmc", [0, 1, 2], [0], stat_type="variance")

    @slow_test
    def test_gmf_variance_variance_stat(self) -> None:
        """Compare GMF variance formula with variance statistic."""
        self._check_variance_formula(
            "gmf", [0, 1], [0], stat_type="variance", recursion_index=[0]
        )

    @slow_test
    def test_grd_variance_variance_stat(self) -> None:
        """Compare GRD variance formula with variance statistic."""
        self._check_variance_formula(
            "grd", [0, 1, 2], [0, 1], stat_type="variance",
            recursion_index=[0, 1], target_cost=200.0
        )

    # Mean-variance statistic tests
    @slow_test
    def test_gmf_variance_mean_variance_stat(self) -> None:
        """Compare GMF variance formula with mean_variance statistic."""
        self._check_variance_formula(
            "gmf", [0, 1], [0], stat_type="mean_variance", recursion_index=[0]
        )

    @slow_test
    def test_mlmc_variance_mean_variance_stat(self) -> None:
        """Compare MLMC variance formula with mean_variance statistic."""
        self._check_variance_formula(
            "mlmc", [0, 1, 2], [0], stat_type="mean_variance"
        )

    # Multi-QoI tests
    @slow_test
    def test_mfmc_variance_mean_multi_qoi(self) -> None:
        """Compare MFMC variance formula with mean statistic and 2 QoIs."""
        self._check_variance_formula("mfmc", [0, 1, 2], [0, 1], stat_type="mean")

    @slow_test
    def test_grd_variance_mean_multi_qoi(self) -> None:
        """Compare GRD variance formula with mean statistic and 2 QoIs."""
        self._check_variance_formula(
            "grd", [0, 1, 2], [0, 1], stat_type="mean", recursion_index=[0, 0]
        )


if __name__ == "__main__":
    unittest.main()
