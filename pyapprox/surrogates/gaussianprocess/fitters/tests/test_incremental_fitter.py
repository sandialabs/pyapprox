"""Tests for GPIncrementalFitter."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFitResult,
    GPFixedHyperparameterFitter,
    GPIncrementalFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestGPIncrementalFitter:
    @pytest.fixture(autouse=True)
    def _setup(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.n_initial = 10
        self.n_test = 5

        X_np = np.random.randn(self.nvars, self.n_initial + 1)
        y_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        self.X_initial = bkd.array(X_np[:, : self.n_initial])
        self.y_initial = bkd.array(y_np[:, : self.n_initial])
        self.X_all = bkd.array(X_np[:, : self.n_initial + 1])
        self.y_all = bkd.array(y_np[:, : self.n_initial + 1])

        X_test_np = np.random.randn(self.nvars, self.n_test)
        self.X_test = bkd.array(X_test_np)

    def _make_gp(self, bkd):
        kernel = Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd)
        return ExactGaussianProcess(kernel, self.nvars, bkd, nugget=0.1)

    def test_fallback_when_no_prev_gp(self, bkd) -> None:
        """First fit (no prev_gp) works via full Cholesky."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)
        result = fitter.fit(gp, self.X_initial, self.y_initial)

        assert isinstance(result, GPFitResult)
        assert result.surrogate().is_fitted()
        mean = result.surrogate().predict(self.X_test)
        assert mean.shape == (1, self.n_test)

    def test_incremental_predictions_match_full(self, bkd) -> None:
        """Predictions from incremental update match full refit."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        # Fit initial data
        result_initial = fitter.fit(gp, self.X_initial, self.y_initial)
        prev_gp = result_initial.surrogate()

        # Incremental fit with one new point
        result_inc = fitter.fit(gp, self.X_all, self.y_all, prev_gp)

        # Full refit for reference
        full_fitter = GPFixedHyperparameterFitter(bkd)
        result_full = full_fitter.fit(gp, self.X_all, self.y_all)

        # Compare predictions
        mean_inc = result_inc.surrogate().predict(self.X_test)
        mean_full = result_full.surrogate().predict(self.X_test)
        bkd.assert_allclose(mean_inc, mean_full, rtol=1e-10)

        std_inc = result_inc.surrogate().predict_std(self.X_test)
        std_full = result_full.surrogate().predict_std(self.X_test)
        bkd.assert_allclose(std_inc, std_full, rtol=1e-10)

    def test_incremental_cholesky_matches_full(self, bkd) -> None:
        """Cholesky factor from incremental update matches full."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        result_initial = fitter.fit(gp, self.X_initial, self.y_initial)
        prev_gp = result_initial.surrogate()

        result_inc = fitter.fit(gp, self.X_all, self.y_all, prev_gp)

        full_fitter = GPFixedHyperparameterFitter(bkd)
        result_full = full_fitter.fit(gp, self.X_all, self.y_all)

        L_inc = result_inc.surrogate().cholesky().factor()
        L_full = result_full.surrogate().cholesky().factor()
        bkd.assert_allclose(L_inc, L_full, rtol=1e-10)

    def test_incremental_alpha_matches_full(self, bkd) -> None:
        """Alpha from incremental update matches full."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        result_initial = fitter.fit(gp, self.X_initial, self.y_initial)
        prev_gp = result_initial.surrogate()

        result_inc = fitter.fit(gp, self.X_all, self.y_all, prev_gp)

        full_fitter = GPFixedHyperparameterFitter(bkd)
        result_full = full_fitter.fit(gp, self.X_all, self.y_all)

        alpha_inc = result_inc.surrogate().alpha()
        alpha_full = result_full.surrogate().alpha()
        bkd.assert_allclose(alpha_inc, alpha_full, rtol=1e-10)

    def test_incremental_nll_matches_full(self, bkd) -> None:
        """NLL from incremental update matches full."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        result_initial = fitter.fit(gp, self.X_initial, self.y_initial)
        prev_gp = result_initial.surrogate()

        result_inc = fitter.fit(gp, self.X_all, self.y_all, prev_gp)

        full_fitter = GPFixedHyperparameterFitter(bkd)
        result_full = full_fitter.fit(gp, self.X_all, self.y_all)

        bkd.assert_allclose(
            bkd.asarray([result_inc.neg_log_marginal_likelihood()]),
            bkd.asarray([result_full.neg_log_marginal_likelihood()]),
            rtol=1e-10,
        )

    def test_fallback_when_multiple_new_points(self, bkd) -> None:
        """>1 new points triggers full Cholesky, still correct."""
        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        # Fit with 5 initial points
        result_initial = fitter.fit(
            gp,
            self.X_initial[:, :5],
            self.y_initial[:, :5],
        )
        prev_gp = result_initial.surrogate()

        # Try to fit with 8 points (3 new) — should fall back
        result = fitter.fit(
            gp,
            self.X_initial[:, :8],
            self.y_initial[:, :8],
            prev_gp,
        )

        assert isinstance(result, GPFitResult)
        assert result.surrogate().is_fitted()
        mean = result.surrogate().predict(self.X_test)
        assert mean.shape == (1, self.n_test)

    def test_multi_qoi(self, bkd) -> None:
        """Incremental update works for nqoi=2."""
        np.random.seed(42)
        X_np = np.random.randn(self.nvars, self.n_initial + 1)
        y_np = np.vstack([
            np.sin(X_np[0, :] + X_np[1, :]),
            np.cos(X_np[0, :] - X_np[1, :]),
        ])  # (2, n_initial + 1)

        X_initial = bkd.array(X_np[:, : self.n_initial])
        y_initial = bkd.array(y_np[:, : self.n_initial])
        X_all = bkd.array(X_np)
        y_all = bkd.array(y_np)

        gp = self._make_gp(bkd)
        fitter = GPIncrementalFitter(bkd)

        result_initial = fitter.fit(gp, X_initial, y_initial)
        prev_gp = result_initial.surrogate()

        result_inc = fitter.fit(gp, X_all, y_all, prev_gp)

        full_fitter = GPFixedHyperparameterFitter(bkd)
        result_full = full_fitter.fit(gp, X_all, y_all)

        mean_inc = result_inc.surrogate().predict(self.X_test)
        mean_full = result_full.surrogate().predict(self.X_test)
        assert mean_inc.shape == (2, self.n_test)
        bkd.assert_allclose(mean_inc, mean_full, rtol=1e-10)

    def test_bkd_accessor(self, bkd) -> None:
        """Fitter bkd() returns the backend."""
        fitter = GPIncrementalFitter(bkd)
        assert fitter.bkd() is bkd
