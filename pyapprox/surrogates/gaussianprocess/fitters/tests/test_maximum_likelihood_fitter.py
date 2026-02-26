"""Tests for GPMaximumLikelihoodFitter."""

import numpy as np

from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    GPMaximumLikelihoodFitter,
    GPOptimizedFitResult,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


class TestGPMaximumLikelihoodFitter:
    """Base test class for GPMaximumLikelihoodFitter."""

    def _setup_data(self, bkd):
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

    def _make_gp(self, bkd, fixed=False):
        kernel = Matern52Kernel(
            [1.0, 1.0],
            (0.1, 10.0),
            self.nvars,
            bkd,
            fixed=fixed,
        )
        return ExactGaussianProcess(kernel, self.nvars, bkd, nugget=0.1)

    def test_returns_optimized_result(self, bkd) -> None:
        """Fitter returns GPOptimizedFitResult."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert isinstance(result, GPOptimizedFitResult)

    def test_fitted_gp_can_predict(self, bkd) -> None:
        """Result surrogate can produce predictions."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        mean = fitted.predict(self.X_test)
        assert mean.shape == (1, self.n_test)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_original_gp_not_modified(self, bkd) -> None:
        """Original GP must NOT be modified by fitter."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        hyps_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        fitter = GPMaximumLikelihoodFitter(bkd)
        fitter.fit(gp, self.X_train, self.y_train)

        assert not gp.is_fitted()
        hyps_after = bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)

    def test_hyperparameters_change_after_optimization(self, bkd) -> None:
        """Optimized hyperparameters should differ from initial."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        initial = bkd.to_numpy(result.initial_hyperparameters())
        optimized = bkd.to_numpy(result.optimized_hyperparameters())
        # They should differ (optimizer changes them)
        assert not np.allclose(initial, optimized, atol=1e-10)

    def test_nll_is_finite(self, bkd) -> None:
        """NLL should be finite."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        nll = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_optimization_result_present(self, bkd) -> None:
        """Optimization result should be present when params are active."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert result.optimization_result() is not None

    def test_no_active_params_skips_optimization(self, bkd) -> None:
        """When all params are fixed, optimization is skipped."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, fixed=True)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        assert result.optimization_result() is None
        # Initial and optimized should be the same
        initial = bkd.to_numpy(result.initial_hyperparameters())
        optimized = bkd.to_numpy(result.optimized_hyperparameters())
        np.testing.assert_array_equal(initial, optimized)

    def test_result_callable(self, bkd) -> None:
        """Result should be callable."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = GPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        pred = result(self.X_test)
        assert pred.shape == (1, self.n_test)

    def test_custom_optimizer(self, bkd) -> None:
        """Fitter accepts custom optimizer."""
        from pyapprox.optimization.minimize.scipy.trust_constr import (
            ScipyTrustConstrOptimizer,
        )

        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        optimizer = ScipyTrustConstrOptimizer(verbosity=0, maxiter=50)
        fitter = GPMaximumLikelihoodFitter(bkd, optimizer=optimizer)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert result.surrogate().is_fitted()
