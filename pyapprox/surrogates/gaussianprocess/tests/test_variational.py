"""
Tests for VariationalGaussianProcess and TorchVariationalGaussianProcess.

Tests are structured progressively:
1. No optimization (all params inactive)
2. Kernel params active only
3. All params active (kernel + inducing + noise)
"""

import unittest
import math
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
from pyapprox.surrogates.gaussianprocess.exact import (
    ExactGaussianProcess,
)
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _sobol_samples(nvars: int, nsamples: int, lb: float, ub: float):
    """Generate Sobol sequence samples in [lb, ub]^nvars."""
    sampler = qmc.Sobol(d=nvars, scramble=True, seed=42)
    raw = sampler.random(nsamples)  # (nsamples, nvars) in [0,1]
    scaled = lb + (ub - lb) * raw
    return scaled.T  # (nvars, nsamples)


class TestVariationalGP(Generic[Array], unittest.TestCase):
    """Base test class for VariationalGaussianProcess."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()
        self.nvars = 1
        self.n_train = 20
        self.n_inducing = 10

        # Training data: simple quadratic on [-1, 1]
        X_np = np.linspace(-1, 1, self.n_train).reshape(1, -1)
        y_np = (X_np[0, :] ** 2)[None, :]
        self.X_train = self._bkd.array(X_np)
        self.y_train = self._bkd.array(y_np)

        # Inducing points via Sobol
        U_np = _sobol_samples(self.nvars, self.n_inducing, -1.0, 1.0)
        self.U_init = self._bkd.array(U_np)

    def _make_kernel(self, fixed: bool = True) -> Matern52Kernel:
        return Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=self.nvars,
            bkd=self._bkd,
            fixed=fixed,
        )

    def _make_inducing(
        self, noise_std: float = 0.1, fixed: bool = True
    ) -> InducingSamples:
        ind = InducingSamples(
            nvars=self.nvars,
            ninducing_samples=self.n_inducing,
            bkd=self._bkd,
            inducing_samples=self.U_init,
            noise_std=noise_std,
            noise_std_bounds=(1e-6, 1.0),
            inducing_sample_bounds=(-2.0, 2.0),
        )
        if fixed:
            ind.hyp_list().set_all_inactive()
        return ind

    def _make_gp(
        self,
        kernel_fixed: bool = True,
        inducing_fixed: bool = True,
        noise_std: float = 0.1,
    ) -> VariationalGaussianProcess:
        kernel = self._make_kernel(fixed=kernel_fixed)
        inducing = self._make_inducing(
            noise_std=noise_std, fixed=inducing_fixed
        )
        return VariationalGaussianProcess(
            kernel=kernel,
            nvars=self.nvars,
            inducing_samples=inducing,
            bkd=self._bkd,
        )

    # ---- Test 1: No optimization ----

    def test_fit_no_optimization(self) -> None:
        """Fit with all params inactive — just computes ELBO and caches."""
        gp = self._make_gp()
        gp.fit(self.X_train, self.y_train)
        self.assertTrue(gp.is_fitted())

        # ELBO should be finite
        neg_elbo = gp.neg_log_marginal_likelihood()
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(neg_elbo))))

    def test_predict_no_optimization(self) -> None:
        """Predict after fit with no optimization."""
        gp = self._make_gp()
        gp.fit(self.X_train, self.y_train)

        mean = gp.predict(self.X_train)
        self.assertEqual(mean.shape, (1, self.n_train))

        std = gp.predict_std(self.X_train)
        self.assertEqual(std.shape, (1, self.n_train))

    def test_call_is_predict(self) -> None:
        """__call__ should be alias for predict."""
        gp = self._make_gp()
        gp.fit(self.X_train, self.y_train)
        self._bkd.assert_allclose(
            gp(self.X_train), gp.predict(self.X_train)
        )

    def test_predict_covariance(self) -> None:
        """Test predict_covariance returns correct shape."""
        gp = self._make_gp()
        gp.fit(self.X_train, self.y_train)
        X_test = self._bkd.array(np.linspace(-1, 1, 5).reshape(1, -1))
        cov = gp.predict_covariance(X_test)
        self.assertEqual(cov.shape, (5, 5))

    def test_alpha_and_cholesky_accessible(self) -> None:
        """alpha() and cholesky() should be available after fit."""
        gp = self._make_gp()
        gp.fit(self.X_train, self.y_train)

        alpha = gp.alpha()
        self.assertEqual(alpha.shape, (1, self.n_train))

        chol = gp.cholesky()
        self.assertEqual(chol.factor().shape, (self.n_train, self.n_train))

    # ---- Test 2: Collapse to exact GP ----

    def test_collapse_to_exact_gp(self) -> None:
        """When inducing = training and noise matches nugget, VGP ≈ exact GP.

        Both GPs predict the latent function f:
        - VGP: noise is separate from kernel, predict_std gives std[f*]
        - Exact GP with nugget: predict_std also gives std[f*]

        The nugget in the exact GP plays the same role as noise_std^2
        in the variational GP: both regularize training via K + sigma^2*I
        but do NOT include noise in predict_std.
        """
        noise_std = 0.1
        noise_var = noise_std ** 2

        # Variational GP with inducing = training points
        kernel_v = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=self.nvars,
            bkd=self._bkd,
            fixed=True,
        )
        ind = InducingSamples(
            nvars=self.nvars,
            ninducing_samples=self.n_train,
            bkd=self._bkd,
            inducing_samples=self.X_train,
            noise_std=noise_std,
            noise_std_bounds=(1e-6, 1.0),
            inducing_sample_bounds=(-2.0, 2.0),
        )
        ind.hyp_list().set_all_inactive()
        vgp = VariationalGaussianProcess(
            kernel=kernel_v,
            nvars=self.nvars,
            inducing_samples=ind,
            bkd=self._bkd,
        )
        vgp.fit(self.X_train, self.y_train)

        # Exact GP with nugget = noise_var (predicts latent f, not noisy y)
        kernel_e = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0),
            nvars=self.nvars,
            bkd=self._bkd,
            fixed=True,
        )
        egp = ExactGaussianProcess(
            kernel=kernel_e,
            nvars=self.nvars,
            bkd=self._bkd,
            nugget=noise_var,
        )
        egp.fit(self.X_train, self.y_train)

        # Compare predictions at test points
        X_test = self._bkd.array(
            np.linspace(-0.9, 0.9, 8).reshape(1, -1)
        )
        mean_v = vgp.predict(X_test)
        mean_e = egp.predict(X_test)
        self._bkd.assert_allclose(mean_v, mean_e, atol=1e-4, rtol=1e-4)

        std_v = vgp.predict_std(X_test)
        std_e = egp.predict_std(X_test)
        self._bkd.assert_allclose(std_v, std_e, atol=1e-3, rtol=1e-3)

    # ---- Test 3: Kernel params active ----

    def test_fit_kernel_active(self) -> None:
        """Fit with kernel params active, inducing fixed."""
        gp = self._make_gp(kernel_fixed=False, inducing_fixed=True)
        gp.fit(self.X_train, self.y_train)
        self.assertTrue(gp.is_fitted())

        # ELBO should be finite after optimization
        neg_elbo = gp.neg_log_marginal_likelihood()
        self.assertTrue(np.isfinite(float(self._bkd.to_numpy(neg_elbo))))

    # ---- Test 4: Input/output transforms ----

    def test_input_output_transforms(self) -> None:
        """Fit with input and output transforms."""
        from pyapprox.surrogates.gaussianprocess.input_transform import (
            InputStandardScaler,
        )
        from pyapprox.surrogates.gaussianprocess.output_transform import (
            OutputStandardScaler,
        )

        gp = self._make_gp()
        in_scaler = InputStandardScaler.from_data(
            self.X_train, self._bkd
        )
        out_scaler = OutputStandardScaler.from_data(
            self.y_train, self._bkd
        )
        gp.fit(
            self.X_train, self.y_train,
            output_transform=out_scaler,
            input_transform=in_scaler,
        )
        self.assertTrue(gp.is_fitted())

        mean = gp.predict(self.X_train)
        self.assertEqual(mean.shape, (1, self.n_train))

    # ---- Test 5: Sparse approximation with optimization ----

    def test_sparse_quadratic_approximation(self) -> None:
        """Optimized VGP with fewer inducing points approximates quadratic.

        Uses 25 inducing points (Sobol) for 50 training points of f(x)=x^2
        on [-1, 1]. Kernel and noise are optimized; inducing locations are
        fixed at Sobol points. The Nyström approximation with 25 inducing
        points limits accuracy to ~1e-3.
        """
        bkd = self._bkd
        nvars = 1
        n_train = 50
        n_inducing = 25

        # Training data: quadratic on [-1, 1]
        X_train = bkd.array(
            np.linspace(-1, 1, n_train).reshape(1, -1)
        )
        y_train = bkd.array(
            (np.linspace(-1, 1, n_train) ** 2).reshape(1, -1)
        )

        # Inducing points via Sobol (fewer than training)
        U_np = _sobol_samples(nvars, n_inducing, -1.0, 1.0)
        U_init = bkd.array(U_np)

        kernel = Matern52Kernel(
            lenscale=[1.0],
            lenscale_bounds=(0.01, 10.0),
            nvars=nvars,
            bkd=bkd,
            fixed=False,
        )
        inducing = InducingSamples(
            nvars=nvars,
            ninducing_samples=n_inducing,
            bkd=bkd,
            inducing_samples=U_init,
            noise_std=1e-2,
            noise_std_bounds=(1e-6, 1.0),
            inducing_sample_bounds=(-2.0, 2.0),
        )
        # Fix inducing locations, optimize kernel + noise
        inducing._inducing_samples.set_all_inactive()

        gp = VariationalGaussianProcess(
            kernel=kernel,
            nvars=nvars,
            inducing_samples=inducing,
            bkd=bkd,
        )
        gp.fit(X_train, y_train)

        # Test on held-out points
        X_test = bkd.array(
            np.linspace(-0.95, 0.95, 50).reshape(1, -1)
        )
        y_test = X_test ** 2
        y_pred = gp.predict(X_test)

        max_err = float(bkd.to_numpy(
            bkd.max(bkd.abs(y_pred - y_test))
        ))
        self.assertLessEqual(
            max_err, 2e-3,
            f"Max prediction error {max_err:.2e} exceeds 2e-3 for quadratic"
        )

    # ---- Test 6: Hyp list structure ----

    def test_hyp_list_combines_all(self) -> None:
        """hyp_list should combine kernel + mean + inducing params."""
        gp = self._make_gp(kernel_fixed=False, inducing_fixed=False)
        hyps = gp.hyp_list()
        # Matern52: 1 lenscale param
        # ZeroMean: 0 params
        # InducingSamples: 1 noise + nvars*n_inducing
        expected = 1 + 0 + 1 + self.nvars * self.n_inducing
        self.assertEqual(hyps.nparams(), expected)


class TestVariationalGPNumpy(TestVariationalGP[NDArray[Any]]):
    """NumPy backend tests."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestVariationalGPTorch(TestVariationalGP[torch.Tensor]):
    """PyTorch backend tests."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestTorchVariationalGP(unittest.TestCase):
    """Torch-specific tests for autograd functionality."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)
        np.random.seed(42)
        self.bkd = TorchBkd()
        self.nvars = 1
        self.n_train = 20
        self.n_inducing = 10

        X_np = np.linspace(-1, 1, self.n_train).reshape(1, -1)
        y_np = (X_np[0, :] ** 2)[None, :]
        self.X_train = self.bkd.array(X_np)
        self.y_train = self.bkd.array(y_np)

        U_np = _sobol_samples(self.nvars, self.n_inducing, -1.0, 1.0)
        self.U_init = self.bkd.array(U_np)

    def _make_torch_vgp(self, kernel_fixed=True, inducing_fixed=True):
        from pyapprox.surrogates.kernels.torch_matern import (
            TorchMaternKernel,
        )
        from pyapprox.surrogates.gaussianprocess.torch_variational import (
            TorchVariationalGaussianProcess,
        )

        kernel = TorchMaternKernel(
            nu=2.5, lenscale=[1.0],
            lenscale_bounds=(0.1, 10.0), nvars=self.nvars,
        )
        if kernel_fixed:
            kernel.hyp_list().set_all_inactive()

        ind = InducingSamples(
            nvars=self.nvars,
            ninducing_samples=self.n_inducing,
            bkd=self.bkd,
            inducing_samples=self.U_init,
            noise_std=0.1,
            noise_std_bounds=(1e-6, 1.0),
            inducing_sample_bounds=(-2.0, 2.0),
        )
        if inducing_fixed:
            ind.hyp_list().set_all_inactive()

        return TorchVariationalGaussianProcess(
            kernel=kernel,
            nvars=self.nvars,
            inducing_samples=ind,
        )

    def test_fit_and_predict(self) -> None:
        """Basic fit and predict with Torch VGP."""
        gp = self._make_torch_vgp()
        gp.fit(self.X_train, self.y_train)
        self.assertTrue(gp.is_fitted())

        mean = gp.predict(self.X_train)
        self.assertEqual(mean.shape, (1, self.n_train))

    def test_autograd_prediction_jacobian(self) -> None:
        """Verify autograd-based prediction Jacobian works."""
        gp = self._make_torch_vgp()
        gp.fit(self.X_train, self.y_train)

        self.assertTrue(hasattr(gp, 'jacobian'))

        sample = self.bkd.array([[0.3]])
        jac = gp.jacobian(sample)
        self.assertEqual(jac.shape, (1, self.nvars))
        self.assertTrue(torch.isfinite(jac).all())

    def test_autograd_loss_jacobian(self) -> None:
        """Verify _configure_loss binds autograd jacobian."""
        from pyapprox.surrogates.gaussianprocess.variational_loss import (
            VariationalGPELBOLoss,
        )

        gp = self._make_torch_vgp(kernel_fixed=False, inducing_fixed=True)
        gp.fit(self.X_train, self.y_train)

        loss = VariationalGPELBOLoss(
            gp, (gp.data().X(), gp.data().y())
        )
        gp._configure_loss(loss)

        self.assertTrue(hasattr(loss, 'jacobian'))

        params = gp.hyp_list().get_active_values()
        jac = loss.jacobian(params)
        self.assertEqual(jac.shape[0], 1)
        self.assertTrue(torch.isfinite(jac).all())

    def test_fit_with_kernel_optimization(self) -> None:
        """Fit with kernel params active uses autograd for optimization."""
        gp = self._make_torch_vgp(kernel_fixed=False, inducing_fixed=True)
        gp.fit(self.X_train, self.y_train)
        self.assertTrue(gp.is_fitted())

        neg_elbo = gp.neg_log_marginal_likelihood()
        self.assertTrue(torch.isfinite(neg_elbo))


if __name__ == "__main__":
    unittest.main()
