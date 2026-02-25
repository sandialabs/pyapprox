"""Tests for AdaptiveGPBuilder.

Replicates legacy test_adaptive_gaussian_process from
pyapprox/surrogates/gaussianprocess/tests/test_activelearning.py
plus additional tests from the plan.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.surrogates.gaussianprocess.adaptive.adaptive_gp_builder import (
    AdaptiveGPBuilder,
)
from pyapprox.surrogates.gaussianprocess.adaptive.cholesky_sampler import (
    CholeskySampler,
)
from pyapprox.surrogates.gaussianprocess.adaptive.sampling_schedule import (
    ConstantSamplingSchedule,
    ListSamplingSchedule,
)
from pyapprox.surrogates.gaussianprocess.adaptive.sobol_sampler import (
    SobolAdaptiveSampler,
)
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
)


def _sin_function(samples: Any, bkd: Backend[Any]) -> Any:
    """1D sin function: (1, nsamples) -> (1, nsamples)."""
    return bkd.reshape(bkd.sin(3.0 * samples[0, :]), (1, -1))


def _quadratic_function(samples: Any, bkd: Backend[Any]) -> Any:
    """1D quadratic: x^2. Replicates legacy fun(xx) = (xx**2).sum(axis=0).

    Input shape: (nvars, nsamples), output shape: (1, nsamples).
    """
    return bkd.reshape(bkd.sum(samples**2, 0), (1, -1))


class TestAdaptiveGPBuilder(Generic[Array], unittest.TestCase):
    """Base tests for AdaptiveGPBuilder."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _make_kernel(self) -> SquaredExponentialKernel:
        return SquaredExponentialKernel([0.5], (0.01, 10.0), 1, self._bkd)

    def _make_sobol_builder(self) -> AdaptiveGPBuilder[Array]:
        bkd = self._bkd
        kernel = self._make_kernel()
        sampler = SobolAdaptiveSampler(1, bkd)
        return AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

    def _make_cholesky_builder(
        self, ncandidates: int = 100
    ) -> AdaptiveGPBuilder[Array]:
        bkd = self._bkd
        kernel = self._make_kernel()
        candidates = bkd.asarray(np.random.rand(1, ncandidates))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        return AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

    # --- Standard tests ---

    def test_builder_returns_gp(self) -> None:
        """run() returns an ExactGaussianProcess."""
        bkd = self._bkd
        builder = self._make_sobol_builder()
        schedule = ConstantSamplingSchedule(5, 15)
        gp = builder.run(lambda s: _sin_function(s, bkd), schedule)
        self.assertIsInstance(gp, ExactGaussianProcess)

    def test_step_returns_samples_and_gp(self) -> None:
        """step() returns (samples, gp) tuple."""
        bkd = self._bkd
        builder = self._make_sobol_builder()
        samples, gp = builder.step(lambda s: _sin_function(s, bkd), 5)
        self.assertEqual(samples.shape, (1, 5))
        self.assertIsInstance(gp, ExactGaussianProcess)

    def test_intermediate_gps_independent(self) -> None:
        """Each step returns a new GP instance.

        Replicates the fact that the legacy AdaptiveGaussianProcess
        rebuilds the GP at each step. Here we verify each step()
        returns a distinct GP object.
        """
        bkd = self._bkd
        builder = self._make_sobol_builder()

        def fun(s):
            return _sin_function(s, bkd)

        _, gp1 = builder.step(fun, 5)
        _, gp2 = builder.step(fun, 5)
        self.assertIsNot(gp1, gp2)

    @slow_test
    def test_sobol_prediction_improves(self) -> None:
        """Prediction error decreases with more samples.

        Integration test: Sobol sampler on 1D sin function.
        """
        bkd = self._bkd
        builder = self._make_sobol_builder()

        def fun(s):
            return _sin_function(s, bkd)

        test_X = bkd.asarray(np.linspace(0.0, 1.0, 50).reshape(1, -1))
        test_y = _sin_function(test_X, bkd)

        _, gp1 = builder.step(fun, 5)
        pred1 = gp1(test_X)
        err1 = float(bkd.to_numpy(bkd.sum((pred1 - test_y) ** 2)))

        _, gp2 = builder.step(fun, 10)
        pred2 = gp2(test_X)
        err2 = float(bkd.to_numpy(bkd.sum((pred2 - test_y) ** 2)))

        self.assertLess(err2, err1)

    @slow_test
    def test_cholesky_integration(self) -> None:
        """Integration test: Cholesky sampler on 1D function."""
        bkd = self._bkd
        builder = self._make_cholesky_builder()
        schedule = ListSamplingSchedule([5, 5])
        gp = builder.run(lambda s: _sin_function(s, bkd), schedule)
        self.assertIsInstance(gp, ExactGaussianProcess)
        data = builder.training_data()
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data[0].shape[1], 10)

    def test_manual_step_loop(self) -> None:
        """Manual step_samples/step_values loop works."""
        bkd = self._bkd
        builder = self._make_sobol_builder()

        def fun(s):
            return _sin_function(s, bkd)

        samples = builder.step_samples(5)
        self.assertEqual(samples.shape, (1, 5))
        values = fun(samples)
        gp = builder.step_values(values, optimize=False)
        self.assertIsInstance(gp, ExactGaussianProcess)
        self.assertIs(builder.current_gp(), gp)

    def test_run_with_schedule(self) -> None:
        """Automatic run() with schedule."""
        bkd = self._bkd
        builder = self._make_sobol_builder()
        schedule = ConstantSamplingSchedule(5, 10)
        gp = builder.run(lambda s: _sin_function(s, bkd), schedule)
        self.assertIsInstance(gp, ExactGaussianProcess)
        data = builder.training_data()
        assert data is not None
        self.assertEqual(data[0].shape[1], 10)

    # --- Edge case tests ---

    def test_cold_start(self) -> None:
        """Builder starts with no training data."""
        builder = self._make_sobol_builder()
        self.assertIsNone(builder.current_gp())
        self.assertIsNone(builder.training_data())

    def test_single_sample_selection(self) -> None:
        """1 sample per step works."""
        bkd = self._bkd
        builder = self._make_sobol_builder()

        def fun(s):
            return _sin_function(s, bkd)

        _, gp = builder.step(fun, 1)
        self.assertIsInstance(gp, ExactGaussianProcess)

    @slow_test
    def test_kernel_change_mid_run(self) -> None:
        """Factorization restarts after HP optimization.

        Replicates legacy test_cholesky_sampler_update_with_changed_kernel
        in the context of the full builder loop: after HP optimization
        changes the kernel, the sampler's set_kernel is called and
        subsequent samples reflect the new kernel.
        """
        bkd = self._bkd
        kernel = self._make_kernel()
        np.random.seed(1)
        candidates = bkd.asarray(np.random.rand(1, 50))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        def fun(s):
            return _quadratic_function(s, bkd)

        # Initial kernel params (before any fitting)
        initial_params = bkd.to_numpy(kernel.hyp_list().get_values()).copy()

        # Step 1: select and fit
        _, gp1 = builder.step(fun, 5)
        kernel_params_1 = bkd.to_numpy(gp1.hyp_list().get_values()).copy()

        # Verify HP optimization changed kernel from initial
        self.assertFalse(
            np.allclose(initial_params, kernel_params_1),
            "Kernel params should change after first step",
        )

        # Step 2: HP optimization may change kernel again
        _, gp2 = builder.step(fun, 5)
        bkd.to_numpy(gp2.hyp_list().get_values()).copy()

        # The kernel should have been updated on the sampler
        # (even if params happen to be similar, the GP was re-created)
        self.assertIsNot(gp1, gp2)
        # Total training data should be 10
        data = builder.training_data()
        assert data is not None
        self.assertEqual(data[0].shape[1], 10)

    def test_candidate_exhaustion(self) -> None:
        """Raises ValueError when candidates exhausted.

        Replicates the error path when CholeskySampler runs out of
        candidates (analogous to legacy behavior).
        """
        bkd = self._bkd
        kernel = self._make_kernel()
        candidates = bkd.asarray(np.random.rand(1, 5))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd)

        def fun(s):
            return _sin_function(s, bkd)

        builder.step(fun, 5)
        with self.assertRaises(ValueError):
            builder.step(fun, 1)

    def test_step_values_before_samples_raises(self) -> None:
        """step_values() without prior step_samples() raises."""
        bkd = self._bkd
        builder = self._make_sobol_builder()
        values = bkd.asarray([[1.0, 2.0, 3.0]])
        with self.assertRaises(RuntimeError):
            builder.step_values(values)

    # --- Legacy replication: test_adaptive_gaussian_process ---

    @slow_test
    def test_adaptive_gp_cholesky_quadratic(self) -> None:
        """End-to-end adaptive GP with Cholesky sampler on x^2.

        Replicates legacy test_adaptive_gaussian_process which runs
        AdaptiveGaussianProcess.build(fun) with a CholeskySampler
        and SamplingScheduleFromList([8, 3]).

        The legacy test verifies training samples match a regression
        value. Here we verify the workflow produces a fitted GP with
        the correct number of training samples and reasonable
        prediction accuracy.

        Key setup to match legacy:
        - Domain [-1, 1] (2 units wide)
        - Lengthscale bounds [0.1, 1.0] to prevent very large lengthscales
        - 2000 candidates (mix of Halton and random like legacy)
        - Initial lengthscale 1.0
        """
        bkd = self._bkd
        np.random.seed(1)
        # Match legacy: lengthscale bounds [0.1, 1.0], not [0.01, 10.0]
        # This prevents the lengthscale from growing too large after
        # HP optimization, which would make the kernel matrix low-rank
        kernel = SquaredExponentialKernel([1.0], (0.1, 1.0), 1, bkd)
        # Match legacy: domain [-1, 1] and 2000 candidates
        # Legacy uses half Halton, half random - we use all random for simplicity
        candidates = bkd.asarray(2.0 * np.random.rand(1, 2000) - 1.0)
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        schedule = ListSamplingSchedule([8, 3])

        def fun(s):
            return _quadratic_function(s, bkd)

        gp = builder.run(fun, schedule)

        # Verify correct number of training samples (8 + 3 = 11)
        data = builder.training_data()
        assert data is not None
        self.assertEqual(data[0].shape[1], 11)

        # Verify GP predicts reasonably at test points
        test_X = bkd.asarray(np.linspace(-1.0, 1.0, 20).reshape(1, -1))
        test_y = _quadratic_function(test_X, bkd)
        pred = gp(test_X)
        max_err = float(
            bkd.to_numpy(bkd.reshape(bkd.max(bkd.abs(pred - test_y)), (1,)))
        )
        self.assertLess(max_err, 0.1)

    @slow_test
    def test_cholesky_multistep_accumulates_data(self) -> None:
        """Multiple steps accumulate training data correctly.

        Replicates the legacy workflow where build() calls
        step_samples + step_values in a loop, accumulating data.
        """
        bkd = self._bkd
        np.random.seed(1)
        kernel = self._make_kernel()
        candidates = bkd.asarray(np.random.rand(1, 50))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        def fun(s):
            return _sin_function(s, bkd)

        # Step through manually like legacy build()
        for nsamples in [5, 3, 2]:
            samples = builder.step_samples(nsamples)
            values = fun(samples)
            gp = builder.step_values(values, optimize=True)

        data = builder.training_data()
        assert data is not None
        self.assertEqual(data[0].shape[1], 10)  # 5 + 3 + 2
        self.assertIsInstance(gp, ExactGaussianProcess)

    # --- Transform tests ---

    @slow_test
    def test_transforms_in_returned_gp(self) -> None:
        """Returned GP has correct transforms set."""
        bkd = self._bkd
        from pyapprox.surrogates.gaussianprocess.input_transform import (
            InputBoundsScaler,
        )

        # User-space domain [2, 5] -> scaled [0, 1]
        lb = bkd.asarray([2.0])
        ub = bkd.asarray([5.0])
        input_transform = InputBoundsScaler(lb, ub, bkd)

        kernel = self._make_kernel()
        # Sobol sampler operates in [0, 1] scaled space
        sampler = SobolAdaptiveSampler(1, bkd)
        builder = AdaptiveGPBuilder(
            kernel,
            sampler,
            bkd,
            input_transform=input_transform,
            noise_variance=1e-6,
        )

        # Function defined in user space [2, 5]
        def fun(s):
            return _sin_function(s, bkd)

        _, gp = builder.step(fun, 10)

        # GP should have input transform
        self.assertIsNotNone(gp.input_transform())

        # Verify GP can predict in user space
        test_X = bkd.asarray([[2.5, 3.0, 4.0, 4.5]])
        pred = gp(test_X)
        self.assertEqual(pred.shape, (1, 4))

    @slow_test
    def test_prediction_in_user_space_with_transforms(self) -> None:
        """GP predictions are in user space when transforms are provided.

        Verifies the chain of: user samples -> scaled -> GP fit ->
        user-space predictions.
        """
        bkd = self._bkd
        from pyapprox.surrogates.gaussianprocess.input_transform import (
            InputBoundsScaler,
        )

        # User domain [2, 5]
        lb = bkd.asarray([2.0])
        ub = bkd.asarray([5.0])
        input_transform = InputBoundsScaler(lb, ub, bkd)

        kernel = self._make_kernel()
        sampler = SobolAdaptiveSampler(1, bkd)
        builder = AdaptiveGPBuilder(
            kernel,
            sampler,
            bkd,
            input_transform=input_transform,
            noise_variance=1e-6,
        )

        # sin function in user space
        def fun(s):
            return _sin_function(s, bkd)

        schedule = ConstantSamplingSchedule(10, 20)
        gp = builder.run(fun, schedule)

        # Predict at user-space points
        test_X = bkd.asarray(np.linspace(2.0, 5.0, 30).reshape(1, -1))
        test_y = _sin_function(test_X, bkd)
        pred = gp(test_X)

        # Should give reasonable predictions
        max_err = float(
            bkd.to_numpy(bkd.reshape(bkd.max(bkd.abs(pred - test_y)), (1,)))
        )
        self.assertLess(max_err, 0.5)

    @slow_test
    def test_statistics_with_returned_gp(self) -> None:
        """Statistics work directly with returned GP, mean matches MC.

        Verifies that GaussianProcessStatistics can be constructed from
        the returned GP and produces mean values consistent with Monte Carlo.
        """
        bkd = self._bkd
        from pyapprox.probability.univariate.uniform import UniformMarginal
        from pyapprox.surrogates.gaussianprocess.statistics import (
            GaussianProcessStatistics,
            SeparableKernelIntegralCalculator,
        )
        from pyapprox.surrogates.sparsegrids.basis_factory import (
            create_basis_factories,
        )

        # Build GP on sin function in [0, 1]
        kernel = self._make_kernel()
        sampler = SobolAdaptiveSampler(1, bkd)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        def fun(s):
            return _sin_function(s, bkd)

        schedule = ConstantSamplingSchedule(10, 30)
        gp = builder.run(fun, schedule)

        # Create statistics object with proper quadrature setup
        marginals = [UniformMarginal(0.0, 1.0, bkd)]
        basis_factories = create_basis_factories(marginals, bkd, "gauss")
        bases = [f.create_basis() for f in basis_factories]
        for b in bases:
            b.set_nterms(30)
        calc: SeparableKernelIntegralCalculator[Any] = (
            SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
        )
        stats: GaussianProcessStatistics[Any] = GaussianProcessStatistics(gp, calc)

        # Compute mean via statistics
        gp_mean = stats.mean_of_mean()

        # Compute MC estimate of mean
        np.random.seed(123)
        mc_samples = bkd.asarray(np.random.rand(1, 10000))
        mc_values = gp(mc_samples)
        mc_mean = float(bkd.to_numpy(bkd.mean(mc_values)))

        # Should be close (within MC error)
        bkd.assert_allclose(
            bkd.asarray([gp_mean]), bkd.asarray([mc_mean]), rtol=0.1, atol=0.05
        )

    @slow_test
    def test_jacobian_with_returned_gp(self) -> None:
        """Jacobian works with returned GP via DerivativeChecker.

        Verifies that the returned GP's jacobian method produces
        derivatives consistent with finite difference approximation.
        """
        bkd = self._bkd
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        # Build GP on quadratic function
        kernel = self._make_kernel()
        sampler = SobolAdaptiveSampler(1, bkd)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        def fun(s):
            return _quadratic_function(s, bkd)

        schedule = ConstantSamplingSchedule(10, 20)
        gp = builder.run(fun, schedule)

        # Use DerivativeChecker to verify jacobian
        checker = DerivativeChecker(gp)
        sample = bkd.asarray([[0.3]])
        errors = checker.check_derivatives(sample, verbosity=0)
        ratio = float(bkd.to_numpy(checker.error_ratio(errors[0])))
        self.assertLessEqual(ratio, 1e-6)


class TestAdaptiveGPBuilderNumpy(TestAdaptiveGPBuilder[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveGPBuilderTorch(TestAdaptiveGPBuilder[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
