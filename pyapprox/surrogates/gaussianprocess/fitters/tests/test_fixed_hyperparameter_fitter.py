"""Tests for GPFixedHyperparameterFitter."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFitResult,
    GPFixedHyperparameterFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestGPFixedHyperparameterFitter:
    """Base test class for GPFixedHyperparameterFitter."""

    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.n_train = 15
        self.n_test = 5

        X_np = np.random.randn(self.nvars, self.n_train)
        y_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        self.X_train = bkd.array(X_np)
        self.y_train = bkd.array(y_np)

        X_test_np = np.random.randn(self.nvars, self.n_test)
        self.X_test = bkd.array(X_test_np)

    def _make_gp(self, bkd):
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)
        return ExactGaussianProcess(kernel, self.nvars, bkd, nugget=0.1)

    def test_returns_gp_fit_result(self, bkd) -> None:
        """Fitter returns GPFitResult."""
        gp = self._make_gp(bkd)
        fitter = GPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert isinstance(result, GPFitResult)

    def test_fitted_gp_can_predict(self, bkd) -> None:
        """Result surrogate can produce predictions."""
        gp = self._make_gp(bkd)
        fitter = GPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        mean = fitted.predict(self.X_test)
        assert mean.shape == (1, self.n_test)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_original_gp_not_modified(self, bkd) -> None:
        """Original GP must NOT be modified by fitter."""
        gp = self._make_gp(bkd)
        hyps_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        fitter = GPFixedHyperparameterFitter(bkd)
        fitter.fit(gp, self.X_train, self.y_train)

        assert not gp.is_fitted()
        hyps_after = bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)

    def test_nll_is_finite(self, bkd) -> None:
        """Negative log marginal likelihood should be finite."""
        gp = self._make_gp(bkd)
        fitter = GPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        nll = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_result_callable(self, bkd) -> None:
        """Result should be callable (delegates to surrogate)."""
        gp = self._make_gp(bkd)
        fitter = GPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        pred = result(self.X_test)
        assert pred.shape == (1, self.n_test)

    def test_result_predict_std(self, bkd) -> None:
        """Result predict_std should return positive values."""
        gp = self._make_gp(bkd)
        fitter = GPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        std = result.predict_std(self.X_test)
        assert std.shape == (1, self.n_test)
        assert bkd.all_bool(std > 0)

    def test_bkd_accessor(self, bkd) -> None:
        """Fitter bkd() should return the backend."""
        fitter = GPFixedHyperparameterFitter(bkd)
        assert fitter.bkd() is bkd
