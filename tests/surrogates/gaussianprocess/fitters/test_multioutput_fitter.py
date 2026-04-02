"""Tests for MultiOutputGP fitters."""

import numpy as np

from pyapprox.surrogates.gaussianprocess.fitters import (
    GPFitResult,
    GPOptimizedFitResult,
    MultiOutputGPFixedHyperparameterFitter,
    MultiOutputGPMaximumLikelihoodFitter,
)
from pyapprox.surrogates.gaussianprocess.multioutput import (
    MultiOutputGP,
)
from pyapprox.surrogates.kernels.iid_gaussian_noise import (
    IIDGaussianNoise,
)
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.kernels.multioutput import (
    IndependentMultiOutputKernel,
)
from pyapprox.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.util.backends.torch import TorchBkd


class TestMultiOutputGPFixedFitter:
    """Base test class for MultiOutputGPFixedHyperparameterFitter.

    These tests work with both NumPy and Torch backends since no
    optimization (and thus no autograd) is needed.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 2
        self.n_train = 20

        X_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = bkd.array(X_np)

        y1_np = np.sin(X_np[0, :] + X_np[1, :])[None, :]
        y2_np = np.cos(X_np[0, :] - X_np[1, :])[None, :]
        self.y_train_list = [
            bkd.array(y1_np),
            bkd.array(y2_np),
        ]
        self.X_train_list = [self.X_train] * self.noutputs

        X_test_np = np.random.randn(self.nvars, 5)
        self.X_test = bkd.array(X_test_np)
        self.X_test_list = [self.X_test] * self.noutputs

    def _make_gp(self, bkd, fixed=True):
        kernels = []
        for _ in range(self.noutputs):
            matern = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                bkd,
                fixed=fixed,
            )
            constant = PolynomialScaling(
                [1.0],
                (0.1, 10.0),
                bkd,
                nvars=self.nvars,
                fixed=fixed,
            )
            noise = IIDGaussianNoise(
                0.1,
                (1e-4, 1.0),
                bkd,
                fixed=fixed,
            )
            kernel = constant * matern + noise
            kernels.append(kernel)
        mo_kernel = IndependentMultiOutputKernel(kernels)
        return MultiOutputGP(mo_kernel, nugget=1e-6)

    def test_fixed_returns_gp_fit_result(self, bkd) -> None:
        """Fixed fitter returns GPFitResult."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        assert isinstance(result, GPFitResult)

    def test_fixed_fitted_can_predict(self, bkd) -> None:
        """Fixed fitter result can predict."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        mean_list = fitted.predict(self.X_test_list)
        assert len(mean_list) == self.noutputs
        for mean in mean_list:
            assert mean.shape == (1, 5)

    def test_fixed_original_not_modified(self, bkd) -> None:
        """Original GP must NOT be modified."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        fitter.fit(gp, self.X_train_list, self.y_train_list)
        assert not gp.is_fitted()

    def test_fixed_nll_is_finite(self, bkd) -> None:
        """NLL should be finite."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        nll = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_ml_no_active_params_skips_optimization(self, bkd) -> None:
        """When all params are fixed, ML fitter skips optimization."""
        self._setup_data(bkd)
        gp = self._make_gp(bkd, fixed=True)
        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        assert result.optimization_result() is None


class TestMultiOutputGPMLFitterTorch:
    """Torch-only tests for MultiOutputGPMaximumLikelihoodFitter.

    ML optimization with active hyperparameters requires autograd
    for gradient computation, so these tests only run with Torch.
    """

    def _setup_data(self):
        import torch
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

    def _make_gp(self, fixed=False):
        from pyapprox.surrogates.gaussianprocess.torch_multioutput import (
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
                [1.0],
                (0.1, 10.0),
                self._bkd,
                nvars=self.nvars,
                fixed=fixed,
            )
            noise = IIDGaussianNoise(
                0.1,
                (1e-4, 1.0),
                self._bkd,
                fixed=fixed,
            )
            kernel = constant * matern + noise
            kernels.append(kernel)
        mo_kernel = IndependentMultiOutputKernel(kernels)
        return TorchMultiOutputGP(mo_kernel, nugget=1e-6)

    def test_ml_returns_optimized_result(self) -> None:
        """ML fitter returns GPOptimizedFitResult."""
        self._setup_data()
        gp = self._make_gp()
        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)
        assert isinstance(result, GPOptimizedFitResult)

    def test_ml_fitted_can_predict(self) -> None:
        """ML fitter result can predict."""
        self._setup_data()
        gp = self._make_gp()
        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train_list, self.y_train_list)

        fitted = result.surrogate()
        mean_list = fitted.predict(self.X_test_list)
        assert len(mean_list) == self.noutputs

    def test_ml_original_not_modified(self) -> None:
        """Original GP must NOT be modified by ML fitter."""
        self._setup_data()
        gp = self._make_gp()
        hyps_before = self._bkd.to_numpy(gp.hyp_list().get_values()).copy()

        fitter = MultiOutputGPMaximumLikelihoodFitter(self._bkd)
        fitter.fit(gp, self.X_train_list, self.y_train_list)

        assert not gp.is_fitted()
        hyps_after = self._bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)
