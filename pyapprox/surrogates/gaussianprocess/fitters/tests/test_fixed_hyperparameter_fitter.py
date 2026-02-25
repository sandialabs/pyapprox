"""Tests for GPFixedHyperparameterFitter."""

import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFixedHyperparameterFitter,
    GPFitResult,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestGPFixedHyperparameterFitter(Generic[Array], unittest.TestCase):
    """Base test class for GPFixedHyperparameterFitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()
        self.nvars = 2
        self.n_train = 15
        self.n_test = 5

        X_np = np.random.randn(self.nvars, self.n_train)
        y_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        self.X_train = self._bkd.array(X_np)
        self.y_train = self._bkd.array(y_np)

        X_test_np = np.random.randn(self.nvars, self.n_test)
        self.X_test = self._bkd.array(X_test_np)

    def _make_gp(self) -> ExactGaussianProcess:
        kernel = Matern52Kernel(
            [1.0, 1.0], (0.1, 10.0), self.nvars, self._bkd
        )
        return ExactGaussianProcess(
            kernel, self.nvars, self._bkd, nugget=0.1
        )

    def test_returns_gp_fit_result(self) -> None:
        """Fitter returns GPFitResult."""
        gp = self._make_gp()
        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        self.assertIsInstance(result, GPFitResult)

    def test_fitted_gp_can_predict(self) -> None:
        """Result surrogate can produce predictions."""
        gp = self._make_gp()
        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        self.assertTrue(fitted.is_fitted())

        mean = fitted.predict(self.X_test)
        self.assertEqual(mean.shape, (1, self.n_test))
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(mean))
        )

    def test_original_gp_not_modified(self) -> None:
        """Original GP must NOT be modified by fitter."""
        gp = self._make_gp()
        hyps_before = self._bkd.to_numpy(
            gp.hyp_list().get_values()
        ).copy()

        fitter = GPFixedHyperparameterFitter(self._bkd)
        fitter.fit(gp, self.X_train, self.y_train)

        self.assertFalse(gp.is_fitted())
        hyps_after = self._bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)

    def test_nll_is_finite(self) -> None:
        """Negative log marginal likelihood should be finite."""
        gp = self._make_gp()
        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        nll = result.neg_log_marginal_likelihood()
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(nll))))

    def test_result_callable(self) -> None:
        """Result should be callable (delegates to surrogate)."""
        gp = self._make_gp()
        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        pred = result(self.X_test)
        self.assertEqual(pred.shape, (1, self.n_test))

    def test_result_predict_std(self) -> None:
        """Result predict_std should return positive values."""
        gp = self._make_gp()
        fitter = GPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        std = result.predict_std(self.X_test)
        self.assertEqual(std.shape, (1, self.n_test))
        self.assertTrue(self._bkd.all_bool(std > 0))

    def test_bkd_accessor(self) -> None:
        """Fitter bkd() should return the backend."""
        fitter = GPFixedHyperparameterFitter(self._bkd)
        self.assertIs(fitter.bkd(), self._bkd)


class TestGPFixedHyperparameterFitterNumpy(
    TestGPFixedHyperparameterFitter[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestGPFixedHyperparameterFitterTorch(
    TestGPFixedHyperparameterFitter[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
