"""Tests for GPFitterAdapter, GPFixedFitterAdapter, GPIncrementalFitterAdapter."""

import numpy as np

from pyapprox.optimization.bayesian.fitter_adapter import (
    GPFitterAdapter,
    GPFixedFitterAdapter,
    GPIncrementalFitterAdapter,
)
from pyapprox.surrogates.gaussianprocess.exact import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.fitters.maximum_likelihood_fitter import (
    GPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestGPFitterAdapter:
    def _make_gp(self, bkd, nvars=1):
        kernel = Matern52Kernel(
            [1.0] * nvars, (0.1, 10.0), nvars, bkd
        )
        return ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)

    def _make_data(self, bkd, nvars=1, n_train=10):
        X_np = np.random.uniform(0, 1, (nvars, n_train))
        y_np = np.sin(2 * np.pi * X_np[0:1, :])
        return bkd.array(X_np), bkd.array(y_np)

    def test_fit_returns_fitted_surrogate(self, bkd) -> None:
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        adapter = GPFitterAdapter(fitter)

        fitted = adapter.fit(gp, X, y)
        assert fitted.is_fitted()

    def test_fitted_can_predict(self, bkd) -> None:
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        adapter = GPFitterAdapter(fitter)

        fitted = adapter.fit(gp, X, y)
        X_test = bkd.array([[0.25, 0.5, 0.75]])
        mean = fitted.predict(X_test)
        assert mean.shape == (1, 3)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_fitted_can_predict_std(self, bkd) -> None:
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        adapter = GPFitterAdapter(fitter)

        fitted = adapter.fit(gp, X, y)
        X_test = bkd.array([[0.25, 0.5, 0.75]])
        std = fitted.predict_std(X_test)
        assert std.shape == (1, 3)
        # Std should be non-negative
        std_np = bkd.to_numpy(std)
        assert np.all(std_np >= 0)

    def test_bkd(self, bkd) -> None:
        np.random.seed(42)
        fitter = GPMaximumLikelihoodFitter(bkd)
        adapter = GPFitterAdapter(fitter)
        assert adapter.bkd() is bkd


class TestGPFixedFitterAdapter:
    def _make_gp(self, bkd, nvars=1):
        kernel = Matern52Kernel(
            [1.0] * nvars, (0.1, 10.0), nvars, bkd
        )
        return ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)

    def _make_data(self, bkd, nvars=1, n_train=10):
        X_np = np.random.uniform(0, 1, (nvars, n_train))
        y_np = np.sin(2 * np.pi * X_np[0:1, :])
        return bkd.array(X_np), bkd.array(y_np)

    def test_fit_returns_fitted_surrogate(self, bkd) -> None:
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        adapter = GPFixedFitterAdapter(bkd)

        fitted = adapter.fit(gp, X, y)
        assert fitted.is_fitted()

    def test_fitted_can_predict(self, bkd) -> None:
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        adapter = GPFixedFitterAdapter(bkd)

        fitted = adapter.fit(gp, X, y)
        X_test = bkd.array([[0.25, 0.5, 0.75]])
        mean = fitted.predict(X_test)
        assert mean.shape == (1, 3)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_bkd(self, bkd) -> None:
        adapter = GPFixedFitterAdapter(bkd)
        assert adapter.bkd() is bkd


class TestGPIncrementalFitterAdapter:
    def _make_gp(self, bkd, nvars=1):
        kernel = Matern52Kernel(
            [1.0] * nvars, (0.1, 10.0), nvars, bkd
        )
        return ExactGaussianProcess(kernel, nvars, bkd, nugget=1e-6)

    def _make_data(self, bkd, nvars=1, n_train=10):
        X_np = np.random.uniform(0, 1, (nvars, n_train))
        y_np = np.sin(2 * np.pi * X_np[0:1, :])
        return bkd.array(X_np), bkd.array(y_np)

    def test_first_fit_works(self, bkd) -> None:
        """First fit (no prev_surrogate) returns fitted GP."""
        np.random.seed(42)
        gp = self._make_gp(bkd)
        X, y = self._make_data(bkd)
        adapter = GPIncrementalFitterAdapter(bkd)

        fitted = adapter.fit(gp, X, y)
        assert fitted.is_fitted()
        X_test = bkd.array([[0.25, 0.5, 0.75]])
        mean = fitted.predict(X_test)
        assert mean.shape == (1, 3)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_sequential_fits_use_incremental(self, bkd) -> None:
        """Sequential single-point fits produce correct predictions."""
        np.random.seed(42)
        gp = self._make_gp(bkd)
        adapter = GPIncrementalFitterAdapter(bkd)

        # Fit initial 5 points
        X_np = np.random.uniform(0, 1, (1, 8))
        y_np = np.sin(2 * np.pi * X_np[0:1, :])
        X = bkd.array(X_np)
        y = bkd.array(y_np)

        fitted = adapter.fit(gp, X[:, :5], y[:, :5])
        assert fitted.is_fitted()

        # Add points one at a time
        for n in range(6, 9):
            fitted = adapter.fit(gp, X[:, :n], y[:, :n])
            assert fitted.is_fitted()

        # Compare final result with full refit
        full_adapter = GPFixedFitterAdapter(bkd)
        fitted_full = full_adapter.fit(gp, X, y)

        X_test = bkd.array([[0.25, 0.5, 0.75]])
        bkd.assert_allclose(
            fitted.predict(X_test),
            fitted_full.predict(X_test),
            rtol=1e-10,
        )

    def test_set_prev_surrogate(self, bkd) -> None:
        """set_prev_surrogate seeds cache for incremental update."""
        np.random.seed(42)
        gp = self._make_gp(bkd)
        adapter = GPIncrementalFitterAdapter(bkd)

        # Fit initial data via full fitter
        X, y = self._make_data(bkd, n_train=5)
        full_adapter = GPFixedFitterAdapter(bkd)
        fitted_full = full_adapter.fit(gp, X, y)

        # Seed the incremental adapter
        adapter.set_prev_surrogate(fitted_full)

        # Now add one point — should use incremental path
        X_np = np.random.uniform(0, 1, (1, 1))
        y_np = np.sin(2 * np.pi * X_np[0:1, :])
        X_new = bkd.hstack([X, bkd.array(X_np)])
        y_new = bkd.hstack([y, bkd.array(y_np)])

        fitted = adapter.fit(gp, X_new, y_new)
        assert fitted.is_fitted()

    def test_bkd(self, bkd) -> None:
        adapter = GPIncrementalFitterAdapter(bkd)
        assert adapter.bkd() is bkd
