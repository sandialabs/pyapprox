"""Tests for VariationalGP fitters."""

import numpy as np
from scipy.stats import qmc

from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFitResult,
    GPOptimizedFitResult,
    VariationalGPFixedHyperparameterFitter,
    VariationalGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples,
)
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel


def _sobol_samples(nvars: int, nsamples: int, lb: float, ub: float):
    """Generate Sobol sequence samples in [lb, ub]^nvars."""
    sampler = qmc.Sobol(d=nvars, scramble=True, seed=42)
    raw = sampler.random(nsamples)
    scaled = lb + (ub - lb) * raw
    return scaled.T


class TestVariationalGPFitters:
    """Base test class for variational GP fitters."""

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 1
        self.n_train = 20
        self.n_inducing = 10

        X_np = np.linspace(-1, 1, self.n_train).reshape(1, -1)
        y_np = (X_np[0, :] ** 2)[None, :]
        self.X_train = bkd.array(X_np)
        self.y_train = bkd.array(y_np)

        U_np = _sobol_samples(self.nvars, self.n_inducing, -1.0, 1.0)
        self.U_init = bkd.array(U_np)

        X_test_np = np.linspace(-0.9, 0.9, 5).reshape(1, -1)
        self.X_test = bkd.array(X_test_np)

    def _make_gp(self, bkd, kernel_fixed=True, inducing_fixed=True):
        kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=self.nvars,
            bkd=bkd,
            fixed=kernel_fixed,
        )
        inducing = InducingSamples(
            nvars=self.nvars,
            ninducing_samples=self.n_inducing,
            bkd=bkd,
            inducing_samples=self.U_init,
            noise_std=0.1,
            noise_std_bounds=(1e-6, 1.0),
            inducing_sample_bounds=(-2.0, 2.0),
        )
        if inducing_fixed:
            inducing.hyp_list().set_all_inactive()
        return VariationalGaussianProcess(
            kernel=kernel,
            nvars=self.nvars,
            inducing_samples=inducing,
            bkd=bkd,
        )

    # ---- Fixed hyperparameter fitter tests ----

    def test_fixed_returns_gp_fit_result(self, bkd) -> None:
        """Fixed fitter returns GPFitResult."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = VariationalGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert isinstance(result, GPFitResult)

    def test_fixed_fitted_can_predict(self, bkd) -> None:
        """Fixed fitter result can predict."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = VariationalGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        mean = fitted.predict(self.X_test)
        assert mean.shape == (1, 5)
        assert bkd.all_bool(bkd.isfinite(mean))

    def test_fixed_original_not_modified(self, bkd) -> None:
        """Original GP must NOT be modified."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = VariationalGPFixedHyperparameterFitter(bkd)
        fitter.fit(gp, self.X_train, self.y_train)
        assert not gp.is_fitted()

    def test_fixed_neg_elbo_is_finite(self, bkd) -> None:
        """Negative ELBO should be finite."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = VariationalGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        neg_elbo = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(neg_elbo)))

    # ---- Maximum likelihood fitter tests ----

    def test_ml_returns_optimized_result(self, bkd) -> None:
        """ML fitter returns GPOptimizedFitResult."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        assert isinstance(result, GPOptimizedFitResult)

    def test_ml_fitted_can_predict(self, bkd) -> None:
        """ML fitter result can predict."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        mean = fitted.predict(self.X_test)
        assert mean.shape == (1, 5)

    def test_ml_original_not_modified(self, bkd) -> None:
        """Original GP must NOT be modified by ML fitter."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, kernel_fixed=False)
        hyps_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        fitter = VariationalGPMaximumLikelihoodFitter(bkd)
        fitter.fit(gp, self.X_train, self.y_train)

        assert not gp.is_fitted()
        hyps_after = bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)

    def test_ml_no_active_params_skips_optimization(self, bkd) -> None:
        """When all params are fixed, optimization is skipped."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = VariationalGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        assert result.optimization_result() is None

    def test_ml_hyperparameters_change(self, bkd) -> None:
        """Optimized hyperparameters should differ from initial."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        initial = bkd.to_numpy(result.initial_hyperparameters())
        optimized = bkd.to_numpy(result.optimized_hyperparameters())
        assert not np.allclose(initial, optimized, atol=1e-10)
