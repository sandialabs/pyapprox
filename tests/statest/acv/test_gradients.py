"""Standalone tests for ACV gradient validation using DerivativeChecker.

These tests verify Jacobian and Hessian implementations for ACV objectives
and constraints. All tests must have error_ratio() <= 1e-6.

Tests use typing array convention: (nqoi, nsamples) for outputs.
"""

import numpy as np
import pytest

from pyapprox.benchmarks.functions.multifidelity.polynomial_ensemble import (
    PolynomialEnsemble,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
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
from pyapprox.util.backends.torch import TorchBkd
from tests._helpers.markers import slow_test


class TestACVLogDeterminantObjectiveGradients:
    """Test gradient correctness for ACVLogDeterminantObjective.

    These tests only run with Torch backend because the objectives use
    bkd.jacobian() which requires autograd.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    def _create_estimator(self, est_type: str, nmodels: int = 3, nqoi: int = 1):
        """Create estimator for testing."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(nqoi, self._bkd)
        stat.set_pilot_quantities(cov)

        recursion_index = self._bkd.array([0] * (nmodels - 1), dtype=int)

        if est_type == "gmf":
            return GMFEstimator(stat, costs, recursion_index=recursion_index)
        elif est_type == "gis":
            # GIS requires recursion_index where each element points to parent
            rec_idx = self._bkd.arange(nmodels - 1, dtype=int)
            return GISEstimator(stat, costs, recursion_index=rec_idx)
        elif est_type == "grd":
            rec_idx = self._bkd.arange(nmodels - 1, dtype=int)
            return GRDEstimator(stat, costs, recursion_index=rec_idx)
        elif est_type == "mfmc":
            return MFMCEstimator(stat, costs)
        elif est_type == "mlmc":
            return MLMCEstimator(stat, costs)
        else:
            raise ValueError(f"Unknown estimator type: {est_type}")

    @pytest.mark.parametrize(
        "est_type",
        ["gmf", "grd", "gis", "mfmc", "mlmc"],
        ids=["gmf", "grd", "gis", "mfmc", "mlmc"],
    )
    @slow_test
    def test_objective_jacobian(self, est_type: str) -> None:
        """Test ACVLogDeterminantObjective Jacobian with DerivativeChecker."""
        nmodels = 3
        target_cost = 50.0
        est = self._create_estimator(est_type, nmodels=nmodels)

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        # Use a starting point away from optimum
        partition_ratios = self._bkd.ones((nmodels - 1, 1)) * 2.0

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(partition_ratios, verbosity=0)

        # Check Jacobian accuracy (use 1e-5 tolerance for numerical precision)
        assert float(checker.error_ratio(errors[0])) <= 1e-5

    @pytest.mark.parametrize(
        "est_type,nmodels",
        [
            ("gmf", 3),
            ("gmf", 4),
            ("grd", 3),
            ("mfmc", 3),
        ],
        ids=["gmf_3", "gmf_4", "grd_3", "mfmc_3"],
    )
    @slow_test
    def test_objective_jacobian_different_sizes(
        self, est_type: str, nmodels: int
    ) -> None:
        """Test Jacobian with different model counts."""
        target_cost = 100.0
        est = self._create_estimator(est_type, nmodels=nmodels)

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        partition_ratios = self._bkd.ones((nmodels - 1, 1)) * 3.0

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(partition_ratios, verbosity=0)

        assert float(checker.error_ratio(errors[0])) <= 1e-6


class TestACVPartitionConstraintGradients:
    """Test gradient correctness for ACVPartitionConstraint.

    These tests only run with Torch backend because the constraints use
    bkd.jacobian() which requires autograd.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    def _create_estimator(self, est_type: str, nmodels: int = 3):
        """Create estimator for testing."""
        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        recursion_index = self._bkd.array([0] * (nmodels - 1), dtype=int)

        if est_type == "gmf":
            return GMFEstimator(stat, costs, recursion_index=recursion_index)
        elif est_type == "gis":
            rec_idx = self._bkd.arange(nmodels - 1, dtype=int)
            return GISEstimator(stat, costs, recursion_index=rec_idx)
        elif est_type == "grd":
            rec_idx = self._bkd.arange(nmodels - 1, dtype=int)
            return GRDEstimator(stat, costs, recursion_index=rec_idx)
        elif est_type == "mfmc":
            return MFMCEstimator(stat, costs)
        elif est_type == "mlmc":
            return MLMCEstimator(stat, costs)
        else:
            raise ValueError(f"Unknown estimator type: {est_type}")

    @pytest.mark.parametrize(
        "est_type",
        ["gmf", "grd", "gis", "mfmc", "mlmc"],
        ids=["gmf", "grd", "gis", "mfmc", "mlmc"],
    )
    @slow_test
    def test_constraint_jacobian(self, est_type: str) -> None:
        """Test ACVPartitionConstraint Jacobian with DerivativeChecker."""
        nmodels = 3
        target_cost = 50.0
        est = self._create_estimator(est_type, nmodels=nmodels)

        constraint = ACVPartitionConstraint(est, target_cost)

        # Use a starting point
        partition_ratios = self._bkd.ones((nmodels - 1, 1)) * 2.0

        checker = DerivativeChecker(constraint)
        # Need weights for multi-qoi constraint
        weights = self._bkd.ones((constraint.nqoi(), 1))
        errors = checker.check_derivatives(
            partition_ratios, weights=weights, verbosity=0
        )

        # Check Jacobian accuracy (use 1e-5 tolerance for numerical precision)
        assert float(checker.error_ratio(errors[0])) <= 1e-5


class TestMFMCOptimalSolutionGradients:
    """Test that gradients are zero at MFMC analytical optimal solution."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    @slow_test
    def test_mfmc_gradient_zero_at_optimum(self) -> None:
        """Test that MFMC objective gradient is zero at analytical optimum."""
        nmodels = 3
        target_cost = 50.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        est = MFMCEstimator(stat, costs)

        # Get analytical MFMC solution
        mfmc_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost, self._bkd
        )

        # Convert to partition ratios
        partition_ratios = est._native_ratios_to_npartition_ratios(mfmc_ratios)

        # Create objective
        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        # Gradient should be zero at optimum
        jacobian = objective.jacobian(partition_ratios[:, None])
        expected_zeros = self._bkd.zeros((1, nmodels - 1))

        self._bkd.assert_allclose(jacobian, expected_zeros, atol=1e-8)

    @slow_test
    def test_mfmc_objective_value_at_optimum(self) -> None:
        """Test MFMC objective value matches analytical solution."""
        nmodels = 3
        target_cost = 50.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        est = MFMCEstimator(stat, costs)

        # Get analytical MFMC solution
        mfmc_ratios, mfmc_log_variance = _allocate_samples_mfmc(
            cov, costs, target_cost, self._bkd
        )
        partition_ratios = est._native_ratios_to_npartition_ratios(mfmc_ratios)

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        computed_log_var = objective(partition_ratios[:, None])

        self._bkd.assert_allclose(
            self._bkd.exp(computed_log_var).flatten(),
            self._bkd.atleast_1d(self._bkd.exp(mfmc_log_variance)),
            rtol=1e-10,
        )


class TestMLMCOptimalSolutionGradients:
    """Test that gradients are zero at MLMC analytical optimal solution."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    @slow_test
    def test_mlmc_gradient_zero_at_optimum(self) -> None:
        """Test that MLMC objective gradient is zero at analytical optimum."""
        nmodels = 3
        target_cost = 81.0

        # Use specific covariance and costs for MLMC test
        # These give mlmc with unit variance and level variances [1, 4, 4]
        costs = self._bkd.array([6.0, 3.0, 1.0])
        cov = self._bkd.asarray(
            [[1.00, 0.50, 0.25], [0.50, 1.00, 0.50], [0.25, 0.50, 4.00]]
        )

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        est = MLMCEstimator(stat, costs)

        # Get analytical MLMC solution
        mlmc_ratios, mlmc_log_variance = _allocate_samples_mlmc(
            cov, costs, target_cost, self._bkd
        )

        # Convert to partition ratios
        partition_ratios = est._native_ratios_to_npartition_ratios(mlmc_ratios)

        # MLMC uses suboptimal weights, so we need to test with
        # the same covariance function
        def mlmc_cov(npartition_samples):
            CF, cf = est._get_discrepancy_covariances(npartition_samples)
            weights = est._weights(CF, cf)
            return est._covariance_non_optimal_weights(
                est._stat.high_fidelity_estimator_covariance(npartition_samples[0]),
                weights,
                CF,
                cf,
            )

        original_cov_func = est._covariance_from_npartition_samples
        est._covariance_from_npartition_samples = mlmc_cov

        # Create objective
        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        # Gradient should be zero at optimum
        jacobian = objective.jacobian(partition_ratios[:, None])
        expected_zeros = self._bkd.zeros((1, nmodels - 1))

        # Restore
        est._covariance_from_npartition_samples = original_cov_func

        self._bkd.assert_allclose(jacobian, expected_zeros, atol=1e-8)

    @slow_test
    def test_mlmc_objective_value_at_optimum(self) -> None:
        """Test MLMC objective value matches analytical solution."""
        target_cost = 81.0

        costs = self._bkd.array([6.0, 3.0, 1.0])
        cov = self._bkd.asarray(
            [[1.00, 0.50, 0.25], [0.50, 1.00, 0.50], [0.25, 0.50, 4.00]]
        )

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        est = MLMCEstimator(stat, costs)

        # Get analytical MLMC solution
        mlmc_ratios, mlmc_log_variance = _allocate_samples_mlmc(
            cov, costs, target_cost, self._bkd
        )
        partition_ratios = est._native_ratios_to_npartition_ratios(mlmc_ratios)

        # Set up MLMC-style covariance
        def mlmc_cov(npartition_samples):
            CF, cf = est._get_discrepancy_covariances(npartition_samples)
            weights = est._weights(CF, cf)
            return est._covariance_non_optimal_weights(
                est._stat.high_fidelity_estimator_covariance(npartition_samples[0]),
                weights,
                CF,
                cf,
            )

        original_cov_func = est._covariance_from_npartition_samples
        est._covariance_from_npartition_samples = mlmc_cov

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        computed_log_var = objective(partition_ratios[:, None])

        # Restore
        est._covariance_from_npartition_samples = original_cov_func

        self._bkd.assert_allclose(
            self._bkd.exp(computed_log_var).flatten(),
            self._bkd.atleast_1d(self._bkd.exp(mlmc_log_variance)),
            rtol=1e-10,
        )


class TestDerivativeCheckerConvergence:
    """Test that DerivativeChecker shows proper convergence behavior."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        np.random.seed(42)
        self._bkd = TorchBkd()

    @pytest.mark.parametrize(
        "est_type",
        ["gmf", "grd", "mfmc"],
        ids=["gmf", "grd", "mfmc"],
    )
    @slow_test
    def test_jacobian_fd_convergence(self, est_type: str) -> None:
        """Test that finite difference errors show second-order convergence."""
        nmodels = 3
        target_cost = 50.0

        ensemble = PolynomialEnsemble(self._bkd, nmodels=nmodels)
        cov = ensemble.covariance_matrix()
        costs = ensemble.costs()

        stat = MultiOutputMean(1, self._bkd)
        stat.set_pilot_quantities(cov)

        if est_type == "gmf":
            rec_idx = self._bkd.array([0] * (nmodels - 1), dtype=int)
            est = GMFEstimator(stat, costs, recursion_index=rec_idx)
        elif est_type == "grd":
            rec_idx = self._bkd.arange(nmodels - 1, dtype=int)
            est = GRDEstimator(stat, costs, recursion_index=rec_idx)
        else:
            est = MFMCEstimator(stat, costs)

        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(est)

        partition_ratios = self._bkd.ones((nmodels - 1, 1)) * 2.0

        # Custom fd_eps to test convergence
        fd_eps = self._bkd.flip(self._bkd.logspace(-12, 0, 13))

        checker = DerivativeChecker(objective)
        errors = checker.check_derivatives(partition_ratios, fd_eps=fd_eps, verbosity=0)

        # Error ratio should show second-order convergence (~0.25)
        ratio = checker.error_ratio(errors[0])
        assert float(ratio) <= 2e-6
