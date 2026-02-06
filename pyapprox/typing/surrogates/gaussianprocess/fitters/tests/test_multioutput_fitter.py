"""Tests for MultiOutputGP fitters."""

import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import Matern52Kernel
from pyapprox.typing.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.typing.surrogates.kernels.iid_gaussian_noise import (
    IIDGaussianNoise,
)
from pyapprox.typing.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
)
from pyapprox.typing.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)
from pyapprox.typing.surrogates.gaussianprocess.fitters import (
    MultiOutputGPFixedHyperparameterFitter,
    MultiOutputGPMaximumLikelihoodFitter,
    GPFitResult,
    GPOptimizedFitResult,
)
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


class TestMultiOutputGPFixedFitter(Generic[Array], unittest.TestCase):
    """Base test class for MultiOutputGPFixedHyperparameterFitter.

    These tests work with both NumPy and Torch backends since no
    optimization (and thus no autograd) is needed.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()
        self.nvars = 2
        self.noutputs = 2
        self.n_train = 20

        X_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = self._bkd.array(X_np)

        y1_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        y2_np = np.cos(X_np[0, :] - X_np[1, :])[None, :]
        self.y_train_list = [
            self._bkd.array(y1_np),
            self._bkd.array(y2_np),
        ]
        self.X_train_list = [self.X_train] * self.noutputs

        X_test_np = np.random.randn(self.nvars, 5)
        self.X_test = self._bkd.array(X_test_np)
        self.X_test_list = [self.X_test] * self.noutputs

    def _make_gp(self, fixed: bool = True) -> MultiOutputGP:
        kernels = []
        for _ in range(self.noutputs):
            matern = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self._bkd,
                fixed=fixed,
            )
            constant = PolynomialScaling(
                [1.0], (0.1, 10.0), self._bkd, nvars=self.nvars,
                fixed=fixed,
            )
            noise = IIDGaussianNoise(
                0.1, (1e-4, 1.0), self._bkd, fixed=fixed,
            )
            kernel = constant * matern + noise
            kernels.append(kernel)
        mo_kernel = IndependentMultiOutputKernel(kernels)
        return MultiOutputGP(mo_kernel, nugget=1e-6)

    def test_fixed_returns_gp_fit_result(self) -> None:
        """Fixed fitter returns GPFitResult."""
        gp = self._make_gp()
        fitter = MultiOutputGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        self.assertIsInstance(result, GPFitResult)

    def test_fixed_fitted_can_predict(self) -> None:
        """Fixed fitter result can predict."""
        gp = self._make_gp()
        fitter = MultiOutputGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)

        fitted = result.surrogate()
        self.assertTrue(fitted.is_fitted())

        mean_list = fitted.predict(self.X_test_list)
        self.assertEqual(len(mean_list), self.noutputs)
        for mean in mean_list:
            self.assertEqual(mean.shape, (1, 5))

    def test_fixed_original_not_modified(self) -> None:
        """Original GP must NOT be modified."""
        gp = self._make_gp()
        fitter = MultiOutputGPFixedHyperparameterFitter(self._bkd)
        fitter.fit(gp, self.X_train_list, self.y_train_list)
        self.assertFalse(gp.is_fitted())

    def test_fixed_nll_is_finite(self) -> None:
        """NLL should be finite."""
        gp = self._make_gp()
        fitter = MultiOutputGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        nll = result.neg_log_marginal_likelihood()
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(nll))))

    def test_ml_no_active_params_skips_optimization(self) -> None:
        """When all params are fixed, ML fitter skips optimization."""
        gp = self._make_gp(fixed=True)
        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        self.assertIsNone(result.optimization_result())


class TestMultiOutputGPFixedFitterNumpy(
    TestMultiOutputGPFixedFitter[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMultiOutputGPFixedFitterTorch(
    TestMultiOutputGPFixedFitter[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestMultiOutputGPMLFitterTorch(unittest.TestCase):
    """Torch-only tests for MultiOutputGPMaximumLikelihoodFitter.

    ML optimization with active hyperparameters requires autograd
    for gradient computation, so these tests only run with Torch.
    """

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        np.random.seed(42)
        self._bkd = TorchBkd()
        self.nvars = 2
        self.noutputs = 2
        self.n_train = 20

        X_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = self._bkd.array(X_np)

        y1_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        y2_np = np.cos(X_np[0, :] - X_np[1, :])[None, :]
        self.y_train_list = [
            self._bkd.array(y1_np),
            self._bkd.array(y2_np),
        ]
        self.X_train_list = [self.X_train] * self.noutputs

        X_test_np = np.random.randn(self.nvars, 5)
        self.X_test = self._bkd.array(X_test_np)
        self.X_test_list = [self.X_test] * self.noutputs

    def _make_gp(self, fixed: bool = False) -> MultiOutputGP:
        from pyapprox.typing.surrogates.gaussianprocess.torch_multioutput import (
            TorchMultiOutputGP,
        )
        kernels = []
        for _ in range(self.noutputs):
            matern = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self._bkd,
                fixed=fixed,
            )
            constant = PolynomialScaling(
                [1.0], (0.1, 10.0), self._bkd, nvars=self.nvars,
                fixed=fixed,
            )
            noise = IIDGaussianNoise(
                0.1, (1e-4, 1.0), self._bkd, fixed=fixed,
            )
            kernel = constant * matern + noise
            kernels.append(kernel)
        mo_kernel = IndependentMultiOutputKernel(kernels)
        return TorchMultiOutputGP(mo_kernel, nugget=1e-6)

    def test_ml_returns_optimized_result(self) -> None:
        """ML fitter returns GPOptimizedFitResult."""
        gp = self._make_gp()
        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        self.assertIsInstance(result, GPOptimizedFitResult)

    def test_ml_fitted_can_predict(self) -> None:
        """ML fitter result can predict."""
        gp = self._make_gp()
        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)

        fitted = result.surrogate()
        mean_list = fitted.predict(self.X_test_list)
        self.assertEqual(len(mean_list), self.noutputs)

    def test_ml_original_not_modified(self) -> None:
        """Original GP must NOT be modified by ML fitter."""
        gp = self._make_gp()
        hyps_before = self._bkd.to_numpy(
            gp.hyp_list().get_values()
        ).copy()

        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        fitter.fit(gp, self.X_train_list, self.y_train_list)

        self.assertFalse(gp.is_fitted())
        hyps_after = self._bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
