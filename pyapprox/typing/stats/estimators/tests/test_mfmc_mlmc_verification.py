"""Verification tests for MFMC and MLMC estimators against legacy formulas.

These tests verify that the typing MFMC/MLMC implementations match
the analytical formulas from the legacy implementation.
"""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.stats.statistics.mean import MultiOutputMean
from pyapprox.typing.stats.estimators.acv import MFMCEstimator, MLMCEstimator
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestMFMCVerification(Generic[Array], unittest.TestCase):
    """Tests verifying MFMC against legacy formulas."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_mfmc_allocation_matches_legacy_formula(self):
        """Verify MFMC analytical allocation matches legacy formula."""
        bkd = self._bkd
        target_cost = 10.0
        # Use covariance with decreasing correlations (required for MFMC)
        cov = bkd.asarray([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0]
        ])
        costs = bkd.asarray([4.0, 2.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mfmc = MFMCEstimator(stat, costs, bkd)

        # Get analytical allocation
        nsample_ratios, log_variance = mfmc.allocate_samples_analytical(target_cost)

        # Compute expected using legacy formula directly
        # Step 1: Convert to correlation
        variances = bkd.get_diagonal(cov)
        std_devs = bkd.sqrt(variances)
        corr = cov / bkd.outer(std_devs, std_devs)

        # Step 2: Compute r for each model
        nmodels = 3
        r_expected = []
        for ii in range(nmodels - 1):
            num = costs[0] * (corr[0, ii] ** 2 - corr[0, ii + 1] ** 2)
            den = costs[ii] * (1 - corr[0, 1] ** 2)
            r_expected.append(bkd.sqrt(num / den))

        # Last model: no next correlation
        num = costs[0] * corr[0, -1] ** 2
        den = costs[-1] * (1 - corr[0, 1] ** 2)
        r_expected.append(bkd.sqrt(num / den))
        r = bkd.stack(r_expected)

        # nsample_ratios should be r[1:] (exclude r_0)
        expected_ratios = r[1:]

        bkd.assert_allclose(nsample_ratios, expected_ratios, rtol=1e-10)

    def test_mfmc_rsquared_formula(self):
        """Test _get_rsquared_mfmc matches legacy formula."""
        bkd = self._bkd
        cov = bkd.asarray([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0]
        ])
        costs = bkd.asarray([4.0, 2.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mfmc = MFMCEstimator(stat, costs, bkd)

        # Sample ratios for testing
        nsample_ratios = bkd.asarray([2.5, 4.0])

        # Compute using the method
        rsquared = mfmc._get_rsquared_mfmc(nsample_ratios)

        # Compute expected using legacy formula
        # First term: (r_1 - 1) / r_1 * cov[0,1]^2 / (cov[0,0] * cov[1,1])
        expected = (
            (nsample_ratios[0] - 1)
            / nsample_ratios[0]
            * cov[0, 1]
            / (cov[0, 0] * cov[1, 1])
            * cov[0, 1]
        )
        # Second term (ii=1)
        p1 = (nsample_ratios[1] - nsample_ratios[0]) / (
            nsample_ratios[1] * nsample_ratios[0]
        )
        p1 = p1 * cov[0, 2] / (cov[0, 0] * cov[2, 2]) * cov[0, 2]
        expected = expected + p1

        bkd.assert_allclose(rsquared, expected, rtol=1e-10)

    def test_mfmc_native_ratios_to_partition_ratios(self):
        """Test conversion from model ratios to partition ratios."""
        bkd = self._bkd
        cov = bkd.asarray([[1.0, 0.9, 0.7], [0.9, 1.0, 0.8], [0.7, 0.8, 1.0]])
        costs = bkd.asarray([4.0, 2.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mfmc = MFMCEstimator(stat, costs, bkd)

        # Test conversion: partition_ratios = [r_1 - 1, r_2 - r_1, ...]
        model_ratios = bkd.asarray([2.0, 3.5])

        partition_ratios = mfmc._native_ratios_to_npartition_ratios(model_ratios)

        expected = bkd.asarray([2.0 - 1.0, 3.5 - 2.0])  # [1.0, 1.5]
        bkd.assert_allclose(partition_ratios, expected, rtol=1e-10)

    def test_mfmc_cost_constraint_satisfied(self):
        """Test that MFMC allocation satisfies cost constraint."""
        bkd = self._bkd
        target_cost = 100.0
        cov = bkd.asarray([
            [1.0, 0.95, 0.8],
            [0.95, 1.0, 0.9],
            [0.8, 0.9, 1.0]
        ])
        costs = bkd.asarray([10.0, 2.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mfmc = MFMCEstimator(stat, costs, bkd)

        nsample_ratios, _ = mfmc.allocate_samples_analytical(target_cost)

        # Compute r for all models (including HF r_0)
        variances = bkd.get_diagonal(cov)
        std_devs = bkd.sqrt(variances)
        corr = cov / bkd.outer(std_devs, std_devs)

        r_list = []
        for ii in range(2):  # nmodels - 1
            num = costs[0] * (corr[0, ii] ** 2 - corr[0, ii + 1] ** 2)
            den = costs[ii] * (1 - corr[0, 1] ** 2)
            r_list.append(bkd.sqrt(num / den))
        num = costs[0] * corr[0, -1] ** 2
        den = costs[-1] * (1 - corr[0, 1] ** 2)
        r_list.append(bkd.sqrt(num / den))
        r = bkd.stack(r_list)

        # nhf = target_cost / dot(costs, r)
        nhf = target_cost / bkd.dot(costs, r)

        # Actual cost: nhf * c_0 + sum_m(r_m * nhf * c_m)
        all_r = bkd.concatenate([bkd.asarray([1.0]), nsample_ratios])
        actual_cost = bkd.sum(all_r * nhf * costs)

        bkd.assert_allclose(
            bkd.reshape(actual_cost, (1,)),
            bkd.asarray([target_cost]),
            rtol=1e-8
        )


class TestMFMCVerificationNumpy(TestMFMCVerification[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMFMCVerificationTorch(TestMFMCVerification[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestMLMCVerification(Generic[Array], unittest.TestCase):
    """Tests verifying MLMC against legacy formulas."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self):
        self._bkd = self.bkd()
        np.random.seed(42)

    def test_mlmc_level_variances(self):
        """Verify level variance computation matches legacy."""
        bkd = self._bkd
        # From legacy test_numerical_mlmc_sample_optimization
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])
        costs = bkd.asarray([6.0, 3.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        level_vars = mlmc._compute_level_variances()

        # Legacy convention: var_deltas[l] = Var(f_l - f_{l+1}), var_deltas[-1] = Var(f_{-1})
        # V_0 = Var(f_0 - f_1) = cov[0,0] + cov[1,1] - 2*cov[0,1] = 1 + 1 - 2*0.5 = 1
        # V_1 = Var(f_1 - f_2) = cov[1,1] + cov[2,2] - 2*cov[1,2] = 1 + 4 - 2*0.5 = 4
        # V_2 = Var(f_2) = cov[2,2] = 4
        expected = bkd.asarray([1.0, 4.0, 4.0])

        bkd.assert_allclose(level_vars, expected, rtol=1e-10)

    def test_mlmc_cost_deltas(self):
        """Verify cost delta computation matches legacy."""
        bkd = self._bkd
        costs = bkd.asarray([6.0, 3.0, 1.0])
        cov = bkd.asarray([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 4.0]])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        cost_deltas = mlmc._compute_cost_deltas()

        # Legacy convention: cost_deltas[l] = cost[l] + cost[l+1], cost_deltas[-1] = cost[-1]
        # c_0 = 6+3 = 9, c_1 = 3+1 = 4, c_2 = 1
        expected = bkd.asarray([9.0, 4.0, 1.0])

        bkd.assert_allclose(cost_deltas, expected, rtol=1e-10)

    def test_mlmc_allocation_matches_legacy_formula(self):
        """Verify MLMC analytical allocation matches legacy formula."""
        bkd = self._bkd
        target_cost = 81.0
        costs = bkd.asarray([6.0, 3.0, 1.0])
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        nsample_ratios, log_variance = mlmc.allocate_samples_analytical(target_cost)

        # Compute expected using legacy formula
        # var_deltas = [1, 4, 4] (legacy convention)
        # cost_deltas = [9, 4, 1]
        var_deltas = bkd.asarray([1.0, 4.0, 4.0])
        cost_deltas = bkd.asarray([9.0, 4.0, 1.0])

        # Lagrange multiplier
        var_cost_prods = var_deltas * cost_deltas
        lagrange = target_cost / bkd.sum(bkd.sqrt(var_cost_prods))

        # Samples per delta
        var_cost_ratios = var_deltas / cost_deltas
        nsamples_per_delta = lagrange * bkd.sqrt(var_cost_ratios)

        # Model ratios
        nhf = nsamples_per_delta[0]
        expected_ratios = bkd.asarray([
            (nsamples_per_delta[0] + nsamples_per_delta[1]) / nhf,
            (nsamples_per_delta[1] + nsamples_per_delta[2]) / nhf,
        ])

        bkd.assert_allclose(nsample_ratios, expected_ratios, rtol=1e-10)

    def test_mlmc_rsquared_formula(self):
        """Test _get_rsquared_mlmc matches legacy formula."""
        bkd = self._bkd
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])
        costs = bkd.asarray([6.0, 3.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        # Sample ratios for testing
        nsample_ratios = bkd.asarray([2.0, 2.5])

        # Compute using the method
        rsquared = mlmc._get_rsquared_mlmc(nsample_ratios)

        # Compute expected using legacy formula
        # rhat[0] = 1, rhat[ii] = nsample_ratios[ii-1] - rhat[ii-1]
        rhat_0 = 1.0
        rhat_1 = float(bkd.to_numpy(nsample_ratios[0])) - rhat_0
        rhat_2 = float(bkd.to_numpy(nsample_ratios[1])) - rhat_1
        rhat = bkd.asarray([rhat_0, rhat_1, rhat_2])

        gamma = bkd.asarray(0.0)
        for ii in range(2):  # nmodels - 1
            vardelta = cov[ii, ii] + cov[ii + 1, ii + 1] - 2 * cov[ii, ii + 1]
            gamma = gamma + vardelta / rhat[ii]
        v = cov[2, 2]
        gamma = gamma + v / rhat[-1]
        gamma = gamma / cov[0, 0]
        expected = 1 - gamma

        bkd.assert_allclose(rsquared, expected, rtol=1e-10)

    def test_mlmc_native_ratios_to_partition_ratios(self):
        """Test MLMC-specific conversion from model to partition ratios."""
        bkd = self._bkd
        cov = bkd.asarray([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 4.0]])
        costs = bkd.asarray([6.0, 3.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        # Test MLMC-specific recurrence:
        # p[0] = r_1 - 1
        # p[i] = r_{i+1} - p[i-1]
        model_ratios = bkd.asarray([2.0, 2.5])

        partition_ratios = mlmc._native_ratios_to_npartition_ratios(model_ratios)

        # p[0] = r_1 - 1 = 2.0 - 1 = 1.0
        # p[1] = r_2 - p[0] = 2.5 - 1.0 = 1.5
        expected = bkd.asarray([1.0, 1.5])
        bkd.assert_allclose(partition_ratios, expected, rtol=1e-10)

    def test_mlmc_direction2_min_cost(self):
        """Test MLMC Direction 2: minimize cost for target variance."""
        bkd = self._bkd
        target_variance = 0.1

        costs = bkd.asarray([6.0, 3.0, 1.0])
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        nsamples, total_cost = mlmc.allocate_samples_for_variance(target_variance)

        # Verify achieved variance equals target
        variances = mlmc._compute_level_variances()

        # Convert nsamples (model samples) to nsamples_per_delta (level samples)
        # using the MLMC partition recurrence:
        # p[0] = n[0], p[l] = n[l] - p[l-1] for l > 0
        # This is the same recurrence as _native_ratios_to_npartition_ratios
        nsamples_per_delta_list = []
        prev_p = nsamples[0]
        nsamples_per_delta_list.append(bkd.reshape(prev_p, (1,)))
        for l in range(1, 3):
            curr_p = nsamples[l] - prev_p
            nsamples_per_delta_list.append(bkd.reshape(curr_p, (1,)))
            prev_p = curr_p
        nsamples_per_delta = bkd.concatenate(nsamples_per_delta_list)

        achieved_var = bkd.sum(variances / nsamples_per_delta)

        bkd.assert_allclose(
            bkd.reshape(achieved_var, (1,)),
            bkd.asarray([target_variance]),
            rtol=1e-6
        )

    def test_mlmc_direction2_cost_formula(self):
        """Test that Direction 2 cost follows analytical formula."""
        bkd = self._bkd
        target_variance = 0.05

        costs = bkd.asarray([8.0, 4.0, 1.0])
        cov = bkd.asarray([
            [1.0, 0.6, 0.3],
            [0.6, 1.0, 0.7],
            [0.3, 0.7, 2.0]
        ])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        _, total_cost = mlmc.allocate_samples_for_variance(target_variance)

        # Expected cost: (1/epsilon^2) * (sum sqrt(V_l * c_l))^2
        # where epsilon^2 = target_variance
        variances = mlmc._compute_level_variances()
        cost_deltas = mlmc._compute_cost_deltas()
        sum_sqrt_vc = bkd.sum(bkd.sqrt(variances * cost_deltas))
        expected_cost = (1 / target_variance) * sum_sqrt_vc ** 2

        bkd.assert_allclose(
            bkd.reshape(total_cost, (1,)),
            bkd.reshape(expected_cost, (1,)),
            rtol=1e-10
        )

    def test_mlmc_direction1_direction2_consistency(self):
        """Test that Direction 1 and 2 are consistent inverses."""
        bkd = self._bkd
        target_cost = 100.0

        costs = bkd.asarray([6.0, 3.0, 1.0])
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        # Direction 1: get variance for target_cost
        nsample_ratios, log_variance = mlmc.allocate_samples_analytical(target_cost)
        variance_d1 = bkd.exp(log_variance)

        # Direction 2: get cost for that variance
        nsamples, total_cost_d2 = mlmc.allocate_samples_for_variance(
            float(bkd.to_numpy(variance_d1))
        )

        # Costs should match (within numerical tolerance)
        bkd.assert_allclose(
            bkd.reshape(total_cost_d2, (1,)),
            bkd.asarray([target_cost]),
            rtol=1e-6
        )


class TestMLMCVerificationNumpy(TestMLMCVerification[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        np.random.seed(42)

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestMLMCVerificationTorch(TestMLMCVerification[torch.Tensor]):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestStationaryPointVerification(unittest.TestCase):
    """Tests verifying analytical solutions are stationary points of optimization.

    These tests use PyTorch for autograd support.

    NOTE: These tests will not work until numerical optimization for ACV is
    fully operational. The MFMC/MLMC estimators use native ratio parameterization
    which differs from the general ACV partition ratio structure expected by
    ACVLogDeterminantObjective.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    @unittest.skip("Requires ACV numerical optimization to be fully operational")
    def test_mfmc_jacobian_zero_at_optimum(self):
        """Test MFMC analytical solution has zero gradient."""
        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVLogDeterminantObjective,
        )

        bkd = self._bkd
        target_cost = 100.0

        cov = bkd.asarray([
            [1.0, 0.9, 0.7],
            [0.9, 1.0, 0.8],
            [0.7, 0.8, 1.0]
        ])
        costs = bkd.asarray([10.0, 2.0, 1.0])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mfmc = MFMCEstimator(stat, costs, bkd)

        # Get analytical allocation
        model_ratios, log_variance = mfmc.allocate_samples_analytical(target_cost)

        # Convert to partition ratios for the objective
        partition_ratios = mfmc._native_ratios_to_npartition_ratios(model_ratios)

        # Create objective
        objective = ACVLogDeterminantObjective()
        objective.set_estimator(mfmc)
        objective.set_target_cost(target_cost)

        # Evaluate objective at analytical solution
        # Need to expand partition ratios with partition 0 ratio = 1
        # But objective expects npartitions - 1 inputs
        sample = bkd.reshape(partition_ratios, (-1, 1))

        # Compute Jacobian
        jac = objective.jacobian(sample)

        # Jacobian should be nearly zero at optimum
        jac_norm = float(bkd.to_numpy(bkd.sqrt(bkd.sum(jac ** 2))))
        self.assertLess(
            jac_norm, 1e-4,
            f"Jacobian norm {jac_norm} should be near zero at analytical optimum"
        )

    @unittest.skip("Requires ACV numerical optimization to be fully operational")
    def test_mlmc_jacobian_zero_at_optimum(self):
        """Test MLMC analytical solution has zero gradient."""
        from pyapprox.typing.stats.estimators.acv.optimization import (
            ACVLogDeterminantObjective,
        )

        bkd = self._bkd
        target_cost = 81.0

        costs = bkd.asarray([6.0, 3.0, 1.0])
        cov = bkd.asarray([
            [1.00, 0.50, 0.25],
            [0.50, 1.00, 0.50],
            [0.25, 0.50, 4.00]
        ])

        stat = MultiOutputMean(nqoi=1, bkd=bkd)
        stat.set_pilot_quantities(cov)
        mlmc = MLMCEstimator(stat, costs, bkd)

        # Get analytical allocation
        model_ratios, log_variance = mlmc.allocate_samples_analytical(target_cost)

        # Convert to partition ratios for the objective
        partition_ratios = mlmc._native_ratios_to_npartition_ratios(model_ratios)

        # Create objective
        objective = ACVLogDeterminantObjective()
        objective.set_estimator(mlmc)
        objective.set_target_cost(target_cost)

        # Evaluate Jacobian at analytical solution
        sample = bkd.reshape(partition_ratios, (-1, 1))
        jac = objective.jacobian(sample)

        # Jacobian should be nearly zero at optimum
        jac_norm = float(bkd.to_numpy(bkd.sqrt(bkd.sum(jac ** 2))))
        self.assertLess(
            jac_norm, 1e-4,
            f"Jacobian norm {jac_norm} should be near zero at analytical optimum"
        )


if __name__ == "__main__":
    unittest.main()
