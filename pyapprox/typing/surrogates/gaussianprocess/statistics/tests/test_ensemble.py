"""
Tests for GP ensemble uncertainty quantification.

Tests the GaussianProcessEnsemble class for sampling GP realizations and
computing the distribution of Sobol sensitivity indices.
"""
import math
import unittest
from typing import Generic, Any, List
import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests, slow_test  # noqa: F401
from pyapprox.typing.surrogates.kernels.matern import SquaredExponentialKernel
from pyapprox.typing.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.typing.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics import (
    SeparableKernelIntegralCalculator,
    GaussianProcessStatistics,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.typing.surrogates.gaussianprocess.statistics.ensemble import (
    GaussianProcessEnsemble,
)


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    """Helper to create quadrature bases from marginals."""
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


class TestGaussianProcessEnsemble(Generic[Array], unittest.TestCase):
    """
    Base test class for GaussianProcessEnsemble.

    Derived classes must implement the bkd() method.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create 2D GP with separable product kernel
        k1 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        k2 = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, self._bkd)
        self._kernel = SeparableProductKernel([k1, k2], self._bkd)

        self._gp = ExactGaussianProcess(
            self._kernel, nvars=2, bkd=self._bkd, nugget=1e-6
        )
        # Skip hyperparameter optimization for these tests
        self._gp.hyp_list().set_all_inactive()

        # Training data
        self._n_train = 15
        X_train_np = np.random.rand(2, self._n_train) * 2 - 1  # [-1, 1]^2
        self._X_train = self._bkd.array(X_train_np)
        # Use backend math operations, shape: (nqoi, n_train)
        self._y_train = self._bkd.reshape(
            self._bkd.sin(math.pi * self._X_train[0, :]) *
            self._bkd.cos(math.pi * self._X_train[1, :]),
            (1, -1)
        )

        self._gp.fit(self._X_train, self._y_train)

        # Marginal distributions
        self._marginals = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Create quadrature bases using sparse grid infrastructure
        self._nquad_points = 30
        bases = _create_quadrature_bases(
            self._marginals, self._nquad_points, self._bkd
        )

        # Create calculator, statistics, and sensitivity
        self._calc = SeparableKernelIntegralCalculator(
            self._gp, bases, self._marginals, bkd=self._bkd
        )
        self._stats = GaussianProcessStatistics(self._gp, self._calc)
        self._sens = GaussianProcessSensitivity(self._stats)

        # Create ensemble
        self._ensemble = GaussianProcessEnsemble(
            self._gp, self._sens, sampler=None
        )

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_nvars(self) -> None:
        """Test nvars returns correct number of variables."""
        self.assertEqual(self._ensemble.nvars(), 2)

    def test_sample_realizations_shape(self) -> None:
        """Test sample_realizations returns correct shapes."""
        n_realizations = 10
        n_sample_points = 50

        realizations, sample_points, weights = self._ensemble.sample_realizations(
            n_realizations, n_sample_points, seed=42
        )

        self.assertEqual(realizations.shape, (n_realizations, n_sample_points))
        self.assertEqual(sample_points.shape, (2, n_sample_points))
        self.assertEqual(weights.shape, (n_sample_points,))

    def test_weights_sum_to_one(self) -> None:
        """Test that quadrature weights sum to 1."""
        _, _, weights = self._ensemble.sample_realizations(5, 100, seed=42)

        weight_sum = float(self._bkd.to_numpy(self._bkd.sum(weights)))
        self.assertAlmostEqual(
            weight_sum, 1.0, places=10, msg=f"Weights sum to {weight_sum}, expected 1.0"
        )

    def test_sample_points_in_domain(self) -> None:
        """Test that sample points are within the expected domain [-1, 1]^2."""
        _, sample_points, _ = self._ensemble.sample_realizations(5, 100, seed=42)

        min_val = float(self._bkd.to_numpy(self._bkd.min(sample_points)))
        max_val = float(self._bkd.to_numpy(self._bkd.max(sample_points)))

        self.assertGreaterEqual(min_val, -1.0 - 1e-10)
        self.assertLessEqual(max_val, 1.0 + 1e-10)

    def test_sample_points_differ_from_training(self) -> None:
        """
        Verify sample points differ from training points.

        This is critical because GP variance is zero at training points,
        which would give degenerate statistics.
        """
        _, sample_points, _ = self._ensemble.sample_realizations(5, 100, seed=42)

        # Compute minimum distance from each sample point to any training point
        X_train = self._X_train  # Shape: (2, n_train)

        # For each sample point, compute distance to all training points
        min_distances = []
        for j in range(sample_points.shape[1]):
            sample_j = sample_points[:, j:j+1]  # Shape: (2, 1)
            # Squared distances
            diff = X_train - sample_j  # Shape: (2, n_train)
            sq_dists = self._bkd.sum(diff * diff, axis=0)  # Shape: (n_train,)
            min_dist = float(self._bkd.to_numpy(self._bkd.min(self._bkd.sqrt(sq_dists))))
            min_distances.append(min_dist)

        # At least half of sample points should be at distance > 0.01 from training
        n_far = sum(1 for d in min_distances if d > 0.01)
        self.assertGreater(
            n_far,
            len(min_distances) // 2,
            "Most sample points should differ significantly from training points",
        )

    def test_realizations_have_variance(self) -> None:
        """
        Test that realizations have non-zero variance across samples.

        This verifies we're not at training points (where variance = 0).
        """
        realizations, _, _ = self._ensemble.sample_realizations(10, 100, seed=42)

        # Compute variance across realizations at each sample point
        # realizations shape: (n_realizations, n_sample_points)
        mean_across_realizations = self._bkd.mean(realizations, axis=0)
        var_across_realizations = self._bkd.mean(
            (realizations - mean_across_realizations) ** 2, axis=0
        )

        # Most sample points should have non-zero variance
        var_np = self._bkd.to_numpy(var_across_realizations)
        n_nonzero = np.sum(var_np > 1e-12)
        n_total = len(var_np)

        self.assertGreater(
            n_nonzero,
            n_total * 0.8,
            f"Only {n_nonzero}/{n_total} sample points have non-zero variance",
        )

    def test_compute_sobol_distribution_keys(self) -> None:
        """Test compute_sobol_distribution returns correct keys."""
        S_dist = self._ensemble.compute_sobol_distribution(
            n_realizations=10, n_sample_points=50, seed=42
        )

        self.assertEqual(set(S_dist.keys()), {0, 1})

    def test_compute_sobol_distribution_shape(self) -> None:
        """Test compute_sobol_distribution returns correct shapes."""
        n_realizations = 10
        S_dist = self._ensemble.compute_sobol_distribution(
            n_realizations=n_realizations, n_sample_points=50, seed=42
        )

        for i in range(self._ensemble.nvars()):
            self.assertEqual(S_dist[i].shape, (n_realizations,))

    def test_sobol_distribution_bounded(self) -> None:
        """Test that S_i values are in [0, 1]."""
        S_dist = self._ensemble.compute_sobol_distribution(
            n_realizations=20, n_sample_points=100, seed=42
        )

        for i, S_i in S_dist.items():
            min_val = float(self._bkd.to_numpy(self._bkd.min(S_i)))
            max_val = float(self._bkd.to_numpy(self._bkd.max(S_i)))

            self.assertGreaterEqual(
                min_val, 0.0 - 1e-10, f"S_{i} has min={min_val} < 0"
            )
            self.assertLessEqual(
                max_val, 1.0 + 1e-10, f"S_{i} has max={max_val} > 1"
            )

    def test_reproducibility_with_seed(self) -> None:
        """Test that using the same seed gives identical results."""
        S_dist1 = self._ensemble.compute_sobol_distribution(
            n_realizations=10, n_sample_points=50, seed=42
        )
        S_dist2 = self._ensemble.compute_sobol_distribution(
            n_realizations=10, n_sample_points=50, seed=42
        )

        for i in range(self._ensemble.nvars()):
            self._bkd.assert_allclose(
                S_dist1[i], S_dist2[i], rtol=1e-10,
                err_msg=f"S_{i} not reproducible with same seed"
            )


class TestZeroVarianceAtTraining(Generic[Array], unittest.TestCase):
    """
    Demonstrate why using training points for MC integration fails.

    GP variance is zero at training points (exact interpolation), so all
    realizations have identical values there. This gives artificially zero
    variation in the computed statistics.
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    def test_zero_variance_at_training_points(self) -> None:
        """
        Verify GP posterior variance is zero at training points.

        This demonstrates why we cannot use training points for MC integration.
        """
        bkd = self._bkd

        # Create simple GP
        k = SquaredExponentialKernel([1.0], (0.1, 10.0), 1, bkd)
        gp = ExactGaussianProcess(k, nvars=1, bkd=bkd, nugget=1e-10)

        # Train on a few points
        X_train = bkd.array([[-0.5, 0.0, 0.5]])
        y_train = bkd.array([[1.0, 0.0, -1.0]])
        gp.hyp_list().set_all_inactive()
        gp.fit(X_train, y_train)

        # Predict at training points
        std_at_train = gp.predict_std(X_train)
        std_np = bkd.to_numpy(std_at_train)

        # Variance should be essentially zero at training points
        # (nugget adds small variance, so we allow a looser threshold)
        self.assertTrue(
            np.all(std_np < 1e-4),
            f"GP std at training points should be ~0, got {std_np}",
        )

        # Compare with a non-training point
        X_test = bkd.array([[0.25]])
        std_at_test = gp.predict_std(X_test)
        std_test = float(bkd.to_numpy(std_at_test[0, 0]))

        self.assertGreater(
            std_test, 1e-3,
            f"GP std away from training should be non-zero, got {std_test}",
        )


class TestMCConvergence(Generic[Array], unittest.TestCase):
    """
    Test Monte Carlo convergence properties.

    As n_realizations increases, the sample mean of S_i should converge
    to E[S_i] (computed analytically).
    """

    __test__ = False

    def setUp(self) -> None:
        """Set up test environment."""
        self._bkd = self.bkd()
        np.random.seed(42)

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError

    @slow_test
    def test_mc_convergence_rate_sobol_mean(self) -> None:
        """
        Test that ensemble mean of S_i converges as n_realizations increases.

        The MC error should decrease roughly as O(1/sqrt(n)).
        """
        bkd = self._bkd

        # Create GP
        k1 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        k2 = SquaredExponentialKernel([0.5], (0.1, 10.0), 1, bkd)
        kernel = SeparableProductKernel([k1, k2], bkd)

        gp = ExactGaussianProcess(kernel, nvars=2, bkd=bkd, nugget=1e-6)
        # Skip hyperparameter optimization for this test
        gp.hyp_list().set_all_inactive()

        # Additive function: f = x_1 + x_2
        n_1d = 8
        x1 = bkd.linspace(-1.0, 1.0, n_1d)
        x2 = bkd.linspace(-1.0, 1.0, n_1d)
        X1, X2 = bkd.meshgrid(x1, x2)
        X_train = bkd.vstack([bkd.flatten(X1), bkd.flatten(X2)])
        # Shape: (nqoi, n_train)
        y_train = bkd.reshape(X_train[0, :] + X_train[1, :], (1, -1))
        gp.fit(X_train, y_train)

        # Create sensitivity objects
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]
        bases = _create_quadrature_bases(marginals, 30, bkd)
        calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        stats = GaussianProcessStatistics(gp, calc)
        sens = GaussianProcessSensitivity(stats)

        # Analytical E[S_i]
        main_effects_analytical = sens.main_effect_indices()
        E_S0_analytical = float(bkd.to_numpy(main_effects_analytical[0]))

        # Create ensemble
        ensemble = GaussianProcessEnsemble(gp, sens)

        # Test with increasing n_realizations
        n_sample_points = 200
        errors = []
        n_values = [100, 500]

        for n_real in n_values:
            S_dist = ensemble.compute_sobol_distribution(
                n_realizations=n_real, n_sample_points=n_sample_points, seed=42
            )
            E_S0_mc = float(bkd.to_numpy(bkd.mean(S_dist[0])))
            error = abs(E_S0_mc - E_S0_analytical)
            errors.append(error)

        # Error should decrease with more realizations
        # This is a soft test - just check that we're in a reasonable range
        # MC error decreases as O(1/sqrt(n)), so error ratio should be ~sqrt(n1/n2)
        expected_ratio = np.sqrt(n_values[0] / n_values[1])
        actual_ratio = errors[0] / max(errors[1], 1e-15)

        # Allow significant tolerance since MC is stochastic
        self.assertGreater(
            actual_ratio, expected_ratio * 0.2,
            f"MC error not decreasing as expected: errors={errors}, "
            f"actual_ratio={actual_ratio:.2f}, expected_ratio≈{expected_ratio:.2f}"
        )


# NumPy backend tests
class TestGaussianProcessEnsembleNumpy(
    TestGaussianProcessEnsemble[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestZeroVarianceAtTrainingNumpy(TestZeroVarianceAtTraining[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMCConvergenceNumpy(TestMCConvergence[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestGaussianProcessEnsembleTorch(
    TestGaussianProcessEnsemble[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestZeroVarianceAtTrainingTorch(TestZeroVarianceAtTraining[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMCConvergenceTorch(TestMCConvergence[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
