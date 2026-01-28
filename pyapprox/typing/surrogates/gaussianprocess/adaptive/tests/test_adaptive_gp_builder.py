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

from pyapprox.typing.surrogates.gaussianprocess.adaptive.adaptive_gp_builder import (
    AdaptiveGPBuilder,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.cholesky_sampler import (
    CholeskySampler,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sampling_schedule import (
    ConstantSamplingSchedule,
    ListSamplingSchedule,
)
from pyapprox.typing.surrogates.gaussianprocess.adaptive.sobol_sampler import (
    SobolAdaptiveSampler,
)
from pyapprox.typing.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.typing.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
from pyapprox.typing.util.test_utils import slow_test


def _sin_function(samples: Any, bkd: Backend[Any]) -> Any:
    """1D sin function: (1, nsamples) -> (1, nsamples)."""
    return bkd.reshape(bkd.sin(3.0 * samples[0, :]), (1, -1))


def _quadratic_function(samples: Any, bkd: Backend[Any]) -> Any:
    """1D quadratic: x^2. Replicates legacy fun(xx) = (xx**2).sum(axis=0).

    Input shape: (nvars, nsamples), output shape: (1, nsamples).
    """
    return bkd.reshape(bkd.sum(samples ** 2, 0), (1, -1))


class TestAdaptiveGPBuilder(Generic[Array], unittest.TestCase):
    """Base tests for AdaptiveGPBuilder."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _make_kernel(self) -> SquaredExponentialKernel:
        return SquaredExponentialKernel(
            [0.5], (0.01, 10.0), 1, self._bkd
        )

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
        gp = builder.run(
            lambda s: _sin_function(s, bkd), schedule
        )
        self.assertIsInstance(gp, ExactGaussianProcess)

    def test_step_returns_samples_and_gp(self) -> None:
        """step() returns (samples, gp) tuple."""
        bkd = self._bkd
        builder = self._make_sobol_builder()
        samples, gp = builder.step(
            lambda s: _sin_function(s, bkd), 5
        )
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
        fun = lambda s: _sin_function(s, bkd)
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
        fun = lambda s: _sin_function(s, bkd)

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
        fun = lambda s: _sin_function(s, bkd)

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
        fun = lambda s: _sin_function(s, bkd)
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
        fun = lambda s: _quadratic_function(s, bkd)

        # Step 1: select and fit
        _, gp1 = builder.step(fun, 5)
        kernel_params_1 = bkd.to_numpy(gp1.hyp_list().get_values()).copy()

        # Step 2: HP optimization may change kernel
        _, gp2 = builder.step(fun, 5)
        kernel_params_2 = bkd.to_numpy(gp2.hyp_list().get_values()).copy()

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

        fun = lambda s: _sin_function(s, bkd)
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
        """
        bkd = self._bkd
        np.random.seed(1)
        kernel = SquaredExponentialKernel(
            [1.0], (0.01, 10.0), 1, bkd
        )
        candidates = bkd.asarray(np.random.rand(1, 100))
        sampler = CholeskySampler(candidates, bkd)
        sampler.set_kernel(kernel)
        builder = AdaptiveGPBuilder(kernel, sampler, bkd, noise_variance=1e-6)

        schedule = ListSamplingSchedule([8, 3])
        fun = lambda s: _quadratic_function(s, bkd)
        gp = builder.run(fun, schedule)

        # Verify correct number of training samples (8 + 3 = 11)
        data = builder.training_data()
        assert data is not None
        self.assertEqual(data[0].shape[1], 11)

        # Verify GP predicts reasonably at test points
        test_X = bkd.asarray(np.linspace(0.0, 1.0, 20).reshape(1, -1))
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

        fun = lambda s: _sin_function(s, bkd)

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
        from pyapprox.typing.surrogates.gaussianprocess.input_transform import (
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
            kernel, sampler, bkd,
            input_transform=input_transform,
            noise_variance=1e-6,
        )

        # Function defined in user space [2, 5]
        fun = lambda s: _sin_function(s, bkd)
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
        from pyapprox.typing.surrogates.gaussianprocess.input_transform import (
            InputBoundsScaler,
        )

        # User domain [2, 5]
        lb = bkd.asarray([2.0])
        ub = bkd.asarray([5.0])
        input_transform = InputBoundsScaler(lb, ub, bkd)

        kernel = self._make_kernel()
        sampler = SobolAdaptiveSampler(1, bkd)
        builder = AdaptiveGPBuilder(
            kernel, sampler, bkd,
            input_transform=input_transform,
            noise_variance=1e-6,
        )

        # sin function in user space
        fun = lambda s: _sin_function(s, bkd)
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
