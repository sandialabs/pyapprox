"""Standalone tests for ACV gradient validation using DerivativeChecker.

These tests verify Jacobian implementations for ACV objectives and
constraints. ACV objectives and constraints declare no analytical
jacobian; production composes autograd on an autodiff backend, so the
derivative tests check that composed object on ``autodiff_bkd``. The
objective values at the analytical MFMC and MLMC optima need no
derivatives and run on every backend.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

from typing import Tuple

import numpy as np
import pytest

from pyapprox.interface.functions.autograd import (
    WithAutogradJacobian,
    WithAutogradJacobianConstraint,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.statest.acv.base import ACVEstimator
from pyapprox.statest.acv.optimization import (
    ACVLogDeterminantObjective,
    ACVPartitionConstraint,
)
from pyapprox.statest.acv.variants import (
    GISEstimator,
    GMFEstimator,
    GRDEstimator,
    MFMCEstimator,
    MLMCEstimator,
    _allocate_samples_mfmc,
    _allocate_samples_mlmc,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.statest import PolynomialEnsembleBenchmark
from tests._helpers.markers import slow_test


@pytest.fixture(autouse=True)
def _seed() -> None:
    np.random.seed(42)


def _create_estimator(
    bkd: Backend[Array], est_type: str, nmodels: int = 3, nqoi: int = 1
) -> ACVEstimator[Array]:
    """Estimator of the given type on the polynomial ensemble."""
    bm = PolynomialEnsembleBenchmark(bkd, nmodels=nmodels)
    costs = bm.problem().costs()
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(bm.ensemble_covariance())

    if est_type == "gmf":
        rec_idx = bkd.zeros((nmodels - 1,), dtype=bkd.index_dtype())
        return GMFEstimator(stat, costs, recursion_index=rec_idx)
    if est_type in ("gis", "grd"):
        # each element points to its parent
        rec_idx = bkd.arange(nmodels - 1, dtype=bkd.index_dtype())
        if est_type == "gis":
            return GISEstimator(stat, costs, recursion_index=rec_idx)
        return GRDEstimator(stat, costs, recursion_index=rec_idx)
    if est_type == "mfmc":
        return MFMCEstimator(stat, costs)
    if est_type == "mlmc":
        return MLMCEstimator(stat, costs)
    raise ValueError(f"Unknown estimator type: {est_type}")


def _objective(
    est: ACVEstimator[Array], target_cost: float
) -> ACVLogDeterminantObjective[Array]:
    objective: ACVLogDeterminantObjective[Array] = ACVLogDeterminantObjective()
    objective.set_target_cost(target_cost)
    objective.set_estimator(est)
    return objective


def _objective_jacobian_error_ratio(
    bkd: AutodiffBackend[Array],
    est: ACVEstimator[Array],
    target_cost: float,
    partition_ratios: Array,
    fd_eps: Array | None = None,
) -> float:
    checker = DerivativeChecker(WithAutogradJacobian(_objective(est, target_cost), bkd))
    errors = checker.check_derivatives(partition_ratios, fd_eps=fd_eps, verbosity=0)
    return bkd.to_float(checker.error_ratio(errors[0]))


def _mfmc_optimum(
    bkd: Backend[Array], target_cost: float
) -> Tuple[MFMCEstimator[Array], Array, Array]:
    """MFMC estimator on the 3-model polynomial ensemble, its analytical
    optimal partition ratios and log variance."""
    bm = PolynomialEnsembleBenchmark(bkd, nmodels=3)
    cov = bm.ensemble_covariance()
    costs = bm.problem().costs()
    stat = MultiOutputMean(1, bkd)
    stat.set_pilot_quantities(cov)
    est = MFMCEstimator(stat, costs)
    ratios, log_variance = _allocate_samples_mfmc(cov, costs, target_cost, bkd)
    return est, est._native_ratios_to_npartition_ratios(ratios), log_variance


def _mlmc_optimum(
    bkd: Backend[Array], target_cost: float
) -> Tuple[MLMCEstimator[Array], Array, Array]:
    """MLMC estimator with unit variance and level variances [1, 4, 4], its
    analytical optimal partition ratios and log variance."""
    costs = bkd.array([6.0, 3.0, 1.0])
    cov = bkd.asarray([[1.00, 0.50, 0.25], [0.50, 1.00, 0.50], [0.25, 0.50, 4.00]])
    stat = MultiOutputMean(1, bkd)
    stat.set_pilot_quantities(cov)
    est = MLMCEstimator(stat, costs)
    ratios, log_variance = _allocate_samples_mlmc(cov, costs, target_cost, bkd)
    return est, est._native_ratios_to_npartition_ratios(ratios), log_variance


class TestACVLogDeterminantObjectiveGradients:
    """Gradient correctness for ACVLogDeterminantObjective."""

    @pytest.mark.parametrize("est_type", ["gmf", "grd", "gis", "mfmc", "mlmc"])
    @slow_test
    def test_objective_jacobian(
        self, autodiff_bkd: AutodiffBackend[Array], est_type: str
    ) -> None:
        bkd = autodiff_bkd
        nmodels = 3
        est = _create_estimator(bkd, est_type, nmodels=nmodels)
        # A starting point away from the optimum
        partition_ratios = bkd.ones((nmodels - 1, 1)) * 2.0
        # 1e-5 tolerance for numerical precision
        assert _objective_jacobian_error_ratio(bkd, est, 50.0, partition_ratios) <= 1e-5

    @pytest.mark.parametrize(
        "est_type,nmodels",
        [("gmf", 3), ("gmf", 4), ("grd", 3), ("mfmc", 3)],
        ids=["gmf_3", "gmf_4", "grd_3", "mfmc_3"],
    )
    @slow_test
    def test_objective_jacobian_different_sizes(
        self, autodiff_bkd: AutodiffBackend[Array], est_type: str, nmodels: int
    ) -> None:
        bkd = autodiff_bkd
        est = _create_estimator(bkd, est_type, nmodels=nmodels)
        partition_ratios = bkd.ones((nmodels - 1, 1)) * 3.0
        assert (
            _objective_jacobian_error_ratio(bkd, est, 100.0, partition_ratios) <= 1e-6
        )

    @pytest.mark.parametrize("est_type", ["gmf", "grd", "mfmc"])
    @slow_test
    def test_jacobian_fd_convergence(
        self, autodiff_bkd: AutodiffBackend[Array], est_type: str
    ) -> None:
        """Finite-difference errors converge at second order on a custom
        step sequence."""
        bkd = autodiff_bkd
        nmodels = 3
        est = _create_estimator(bkd, est_type, nmodels=nmodels)
        partition_ratios = bkd.ones((nmodels - 1, 1)) * 2.0
        fd_eps = bkd.flip(bkd.logspace(-12, 0, 13))
        ratio = _objective_jacobian_error_ratio(
            bkd, est, 50.0, partition_ratios, fd_eps=fd_eps
        )
        assert ratio <= 2e-6


class TestACVPartitionConstraintGradients:
    """Gradient correctness for ACVPartitionConstraint."""

    @pytest.mark.parametrize("est_type", ["gmf", "grd", "gis", "mfmc", "mlmc"])
    @slow_test
    def test_constraint_jacobian(
        self, autodiff_bkd: AutodiffBackend[Array], est_type: str
    ) -> None:
        bkd = autodiff_bkd
        nmodels = 3
        est = _create_estimator(bkd, est_type, nmodels=nmodels)
        constraint = ACVPartitionConstraint(est, 50.0)
        partition_ratios = bkd.ones((nmodels - 1, 1)) * 2.0

        # Constraints likewise get their jacobian composed in production.
        checker = DerivativeChecker(WithAutogradJacobianConstraint(constraint, bkd))
        # The constraint declares no hessian, so only its jacobian is
        # checked and no weights are needed.
        errors = checker.check_derivatives(partition_ratios, verbosity=0)
        # 1e-5 tolerance for numerical precision
        assert bkd.to_float(checker.error_ratio(errors[0])) <= 1e-5


class TestAnalyticalOptimaGradients:
    """The objective gradient vanishes at the analytical MFMC and MLMC
    optima, and the objective equals the analytical log variance."""

    @slow_test
    def test_mfmc_gradient_zero_at_optimum(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        bkd = autodiff_bkd
        target_cost = 50.0
        est, partition_ratios, _ = _mfmc_optimum(bkd, target_cost)
        jacobian = (
            WithAutogradJacobian(_objective(est, target_cost), bkd)
            .derivatives()
            .jacobian
        )
        assert jacobian is not None
        bkd.assert_allclose(
            jacobian(partition_ratios[:, None]), bkd.zeros((1, 2)), atol=1e-8
        )

    @slow_test
    def test_mlmc_gradient_zero_at_optimum(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        bkd = autodiff_bkd
        target_cost = 81.0
        est, partition_ratios, _ = _mlmc_optimum(bkd, target_cost)
        jacobian = (
            WithAutogradJacobian(_objective(est, target_cost), bkd)
            .derivatives()
            .jacobian
        )
        assert jacobian is not None
        bkd.assert_allclose(
            jacobian(partition_ratios[:, None]), bkd.zeros((1, 2)), atol=1e-8
        )

    @slow_test
    def test_mfmc_objective_value_at_optimum(self, bkd: Backend[Array]) -> None:
        target_cost = 50.0
        est, partition_ratios, log_variance = _mfmc_optimum(bkd, target_cost)
        computed = _objective(est, target_cost)(partition_ratios[:, None])
        bkd.assert_allclose(
            bkd.flatten(bkd.exp(computed)),
            bkd.atleast_1d(bkd.exp(log_variance)),
            rtol=1e-10,
        )

    @slow_test
    def test_mlmc_objective_value_at_optimum(self, bkd: Backend[Array]) -> None:
        target_cost = 81.0
        est, partition_ratios, log_variance = _mlmc_optimum(bkd, target_cost)
        computed = _objective(est, target_cost)(partition_ratios[:, None])
        bkd.assert_allclose(
            bkd.flatten(bkd.exp(computed)),
            bkd.atleast_1d(bkd.exp(log_variance)),
            rtol=1e-10,
        )
