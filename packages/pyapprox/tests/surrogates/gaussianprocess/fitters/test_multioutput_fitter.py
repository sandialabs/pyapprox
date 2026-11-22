"""Tests for MultiOutputGP fitters."""

from dataclasses import dataclass
from typing import Generic, List

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
from pyapprox.surrogates.kernels.scalings import PolynomialScalingKernel
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend

NVARS = 2
NOUTPUTS = 2
NTRAIN = 20
NTEST = 5


@dataclass(frozen=True)
class _Data(Generic[Array]):
    X_train_list: List[Array]
    y_train_list: List[Array]
    X_test_list: List[Array]


def _data(bkd: Backend[Array]) -> _Data[Array]:
    np.random.seed(42)
    X_np = np.random.randn(NVARS, NTRAIN)
    X_train = bkd.array(X_np)
    y_train_list = [
        bkd.array(np.sin(X_np[0, :] + X_np[1, :])[None, :]),
        bkd.array(np.cos(X_np[0, :] - X_np[1, :])[None, :]),
    ]
    X_test = bkd.array(np.random.randn(NVARS, NTEST))
    return _Data([X_train] * NOUTPUTS, y_train_list, [X_test] * NOUTPUTS)


def _make_gp(bkd: Backend[Array], fixed: bool) -> MultiOutputGP[Array]:
    kernels = []
    for _ in range(NOUTPUTS):
        matern = Matern52Kernel(
            bkd.ones((NVARS,)),
            (0.1, 10.0),
            NVARS,
            bkd,
            fixed=fixed,
        )
        constant = PolynomialScalingKernel(
            [1.0],
            (0.1, 10.0),
            bkd,
            nvars=NVARS,
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


class TestMultiOutputGPFixedFitter:
    """Tests for MultiOutputGPFixedHyperparameterFitter.

    These tests work with both NumPy and Torch backends since no
    optimization (and thus no autograd) is needed.
    """

    def test_fixed_returns_gp_fit_result(self, bkd: Backend[Array]) -> None:
        """Fixed fitter returns GPFitResult."""
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=True)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)
        assert isinstance(result, GPFitResult)

    def test_fixed_fitted_can_predict(self, bkd: Backend[Array]) -> None:
        """Fixed fitter result can predict."""
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=True)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)

        fitted = result.surrogate()
        assert fitted.is_fitted()

        mean_list = fitted.predict(data.X_test_list)
        assert len(mean_list) == NOUTPUTS
        for mean in mean_list:
            assert mean.shape == (1, NTEST)

    def test_fixed_original_not_modified(self, bkd: Backend[Array]) -> None:
        """Original GP must NOT be modified."""
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=True)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        fitter.fit(gp, data.X_train_list, data.y_train_list)
        assert not gp.is_fitted()

    def test_fixed_nll_is_finite(self, bkd: Backend[Array]) -> None:
        """NLL should be finite."""
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=True)
        fitter = MultiOutputGPFixedHyperparameterFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)
        nll = result.neg_log_marginal_likelihood()
        assert np.isfinite(float(bkd.to_numpy(nll)))

    def test_ml_no_active_params_skips_optimization(self, bkd: Backend[Array]) -> None:
        """When all params are fixed, ML fitter skips optimization."""
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=True)
        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)
        assert result.optimization_result() is None


class TestMultiOutputGPMLFitterAutodiff:
    """Tests for MultiOutputGPMaximumLikelihoodFitter with active
    hyperparameters, whose gradients come from autodiff."""

    def test_ml_returns_optimized_result(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """ML fitter returns GPOptimizedFitResult."""
        bkd = autodiff_bkd
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=False)
        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)
        assert isinstance(result, GPOptimizedFitResult)

    def test_ml_fitted_can_predict(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        """ML fitter result can predict."""
        bkd = autodiff_bkd
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=False)
        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        result = fitter.fit(gp, data.X_train_list, data.y_train_list)

        fitted = result.surrogate()
        mean_list = fitted.predict(data.X_test_list)
        assert len(mean_list) == NOUTPUTS

    def test_ml_original_not_modified(
        self, autodiff_bkd: AutodiffBackend[Array]
    ) -> None:
        """Original GP must NOT be modified by ML fitter."""
        bkd = autodiff_bkd
        data = _data(bkd)
        gp = _make_gp(bkd, fixed=False)
        hyps_before = bkd.to_numpy(gp.hyp_list().get_values()).copy()

        fitter = MultiOutputGPMaximumLikelihoodFitter(bkd)
        fitter.fit(gp, data.X_train_list, data.y_train_list)

        assert not gp.is_fitted()
        hyps_after = bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)
