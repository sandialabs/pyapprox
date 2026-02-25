"""Tests for VariationalGP fitters."""

import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.stats import qmc

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.surrogates.kernels.matern import Matern52Kernel
from pyapprox.surrogates.gaussianprocess.variational import (
    VariationalGaussianProcess,
)
from pyapprox.surrogates.gaussianprocess.inducing_samples import (
    InducingSamples,
)
from pyapprox.surrogates.gaussianprocess.fitters import (
    VariationalGPFixedHyperparameterFitter,
    VariationalGPMaximumLikelihoodFitter,
    GPFitResult,
    GPOptimizedFitResult,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _sobol_samples(nvars: int, nsamples: int, lb: float, ub: float):
    """Generate Sobol sequence samples in [lb, ub]^nvars."""
    sampler = qmc.Sobol(d=nvars, scramble=True, seed=42)
    raw = sampler.random(nsamples)
    scaled = lb + (ub - lb) * raw
    return scaled.T


class TestVariationalGPFitters(Generic[Array], unittest.TestCase):
    """Base test class for variational GP fitters."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()
        self.nvars = 1
        self.n_train = 20
        self.n_inducing = 10

        X_np = np.linspace(-1, 1, self.n_train).reshape(1, -1)
        y_np = (X_np[0, :] ** 2)[None, :]
        self.X_train = self._bkd.array(X_np)
        self.y_train = self._bkd.array(y_np)

        U_np = _sobol_samples(self.nvars, self.n_inducing, -1.0, 1.0)
        self.U_init = self._bkd.array(U_np)

        X_test_np = np.linspace(-0.9, 0.9, 5).reshape(1, -1)
        self.X_test = self._bkd.array(X_test_np)

    def _make_gp(
        self, kernel_fixed: bool = True, inducing_fixed: bool = True
    ) -> VariationalGaussianProcess:
        kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=self.nvars,
            bkd=self._bkd,
            fixed=kernel_fixed,
        )
        inducing = InducingSamples(
            nvars=self.nvars,
            ninducing_samples=self.n_inducing,
            bkd=self._bkd,
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
            bkd=self._bkd,
        )

    # ---- Fixed hyperparameter fitter tests ----

    def test_fixed_returns_gp_fit_result(self) -> None:
        """Fixed fitter returns GPFitResult."""
        gp = self._make_gp()
        fitter = VariationalGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        self.assertIsInstance(result, GPFitResult)

    def test_fixed_fitted_can_predict(self) -> None:
        """Fixed fitter result can predict."""
        gp = self._make_gp()
        fitter = VariationalGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        self.assertTrue(fitted.is_fitted())

        mean = fitted.predict(self.X_test)
        self.assertEqual(mean.shape, (1, 5))
        self.assertTrue(
            self._bkd.all_bool(self._bkd.isfinite(mean))
        )

    def test_fixed_original_not_modified(self) -> None:
        """Original GP must NOT be modified."""
        gp = self._make_gp()
        fitter = VariationalGPFixedHyperparameterFitter(self._bkd)
        fitter.fit(gp, self.X_train, self.y_train)
        self.assertFalse(gp.is_fitted())

    def test_fixed_neg_elbo_is_finite(self) -> None:
        """Negative ELBO should be finite."""
        gp = self._make_gp()
        fitter = VariationalGPFixedHyperparameterFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        neg_elbo = result.neg_log_marginal_likelihood()
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(neg_elbo))))

    # ---- Maximum likelihood fitter tests ----

    def test_ml_returns_optimized_result(self) -> None:
        """ML fitter returns GPOptimizedFitResult."""
        gp = self._make_gp(kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)
        self.assertIsInstance(result, GPOptimizedFitResult)

    def test_ml_fitted_can_predict(self) -> None:
        """ML fitter result can predict."""
        gp = self._make_gp(kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        fitted = result.surrogate()
        mean = fitted.predict(self.X_test)
        self.assertEqual(mean.shape, (1, 5))

    def test_ml_original_not_modified(self) -> None:
        """Original GP must NOT be modified by ML fitter."""
        gp = self._make_gp(kernel_fixed=False)
        hyps_before = self._bkd.to_numpy(
            gp.hyp_list().get_values()
        ).copy()

        fitter = VariationalGPMaximumLikelihoodFitter(self._bkd)
        fitter.fit(gp, self.X_train, self.y_train)

        self.assertFalse(gp.is_fitted())
        hyps_after = self._bkd.to_numpy(gp.hyp_list().get_values())
        np.testing.assert_array_equal(hyps_before, hyps_after)

    def test_ml_no_active_params_skips_optimization(self) -> None:
        """When all params are fixed, optimization is skipped."""
        gp = self._make_gp()
        fitter = VariationalGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        self.assertIsNone(result.optimization_result())

    def test_ml_hyperparameters_change(self) -> None:
        """Optimized hyperparameters should differ from initial."""
        gp = self._make_gp(kernel_fixed=False)
        fitter = VariationalGPMaximumLikelihoodFitter(self._bkd)
        result = fitter.fit(gp, self.X_train, self.y_train)

        initial = self._bkd.to_numpy(result.initial_hyperparameters())
        optimized = self._bkd.to_numpy(result.optimized_hyperparameters())
        self.assertFalse(np.allclose(initial, optimized, atol=1e-10))


class TestVariationalGPFittersNumpy(
    TestVariationalGPFitters[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestVariationalGPFittersTorch(
    TestVariationalGPFitters[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
