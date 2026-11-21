"""Tests for SUPN MSE loss functions.

Validates loss value, analytical gradient, and exact HVP using
DerivativeChecker and the backend's autodiff.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.surrogates.supn import SUPNMSELoss, create_supn
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array


class TestSUPNMSELoss:
    """Tests for SUPNMSELoss."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_initialization(self, bkd) -> None:
        """Test loss initialization."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = SUPNMSELoss(supn, samples, values, bkd)
        assert loss.nvars() == supn.nparams()
        assert loss.nqoi() == 1

    def test_loss_shape(self, bkd) -> None:
        """Test loss output shape (1, 1)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        loss = SUPNMSELoss(supn, samples, values, bkd)
        params = supn._flatten_params()
        mse = loss(params)
        assert mse.shape == (1, 1)
        assert float(mse[0, 0]) >= 0

    def test_loss_changes_with_params(self, bkd) -> None:
        """Test that loss changes when parameters change."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        loss = SUPNMSELoss(supn, samples, values, bkd)
        params1 = supn._flatten_params()
        mse1 = loss(params1)

        params2 = params1 + bkd.asarray(np.full(params1.shape, 0.5))
        mse2 = loss(params2)
        assert float(mse1[0, 0]) != float(mse2[0, 0])

    def test_loss_zero_at_target(self, bkd) -> None:
        """Test that loss is zero when SUPN matches target exactly."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = supn(samples)

        loss = SUPNMSELoss(supn, samples, values, bkd)
        mse = loss(params)
        bkd.assert_allclose(mse, bkd.zeros((1, 1)), atol=1e-12)

    def test_jacobian_shape(self, bkd) -> None:
        """Test gradient shape (1, nparams)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        loss = SUPNMSELoss(supn, samples, values, bkd)
        params = supn._flatten_params()
        grad = loss.jacobian(params)
        assert grad.shape == (1, supn.nparams())
        assert bkd.all_bool(bkd.isfinite(grad))

    def test_jacobian_zero_at_minimum(self, bkd) -> None:
        """Test that gradient is zero at the optimal solution."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = supn(samples)

        loss = SUPNMSELoss(supn, samples, values, bkd)
        grad = loss.jacobian(params)
        bkd.assert_allclose(grad, bkd.zeros((1, supn.nparams())), atol=1e-10)

    def test_gradient_derivative_checker(self, bkd) -> None:
        """Test gradient using DerivativeChecker."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = SUPNMSELoss(supn, samples, values, bkd)

        checker = DerivativeChecker(loss)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )
        grad_error_ratio = float(checker.error_ratio(errors[0]))
        assert grad_error_ratio < 1e-6, f"Gradient error ratio {grad_error_ratio:.2e}"
        # DerivativeChecker also checks HVP since loss has hvp method
        assert len(errors) == 2, "Expected HVP check (loss has hvp)"
        hvp_error_ratio = float(checker.error_ratio(errors[1]))
        assert hvp_error_ratio < 1e-6, f"HVP error ratio {hvp_error_ratio:.2e}"

    def test_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test gradient and HVP with nqoi > 1 using DerivativeChecker."""
        nqoi = 2
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(nqoi, nsamples))

        loss = SUPNMSELoss(supn, samples, values, bkd)

        checker = DerivativeChecker(loss)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )
        grad_error_ratio = float(checker.error_ratio(errors[0]))
        assert grad_error_ratio < 1e-6, f"Gradient error ratio {grad_error_ratio:.2e}"
        # Loss nqoi=1 so HVP check should still run
        assert len(errors) == 2, "Expected HVP check (loss has hvp)"
        hvp_error_ratio = float(checker.error_ratio(errors[1]))
        assert hvp_error_ratio < 1e-6, f"HVP error ratio {hvp_error_ratio:.2e}"

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP output shape (P, 1)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        loss = SUPNMSELoss(supn, samples, values, bkd)
        params = supn._flatten_params()
        direction = bkd.asarray(np.random.randn(supn.nparams(), 1))
        hvp = loss.hvp(params, direction)
        assert hvp.shape == (supn.nparams(), 1)


class TestSUPNMSELossAutograd:
    """The analytical gradient and HVP match the backend's autodiff ones,
    which need the computation graph to survive."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_gradient_matches_autodiff(
        self, autodiff_bkd: AutodiffBackend[Array], nqoi: int
    ) -> None:
        bkd = autodiff_bkd
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 15)))
        values = bkd.asarray(np.random.randn(nqoi, 15))
        loss = SUPNMSELoss(supn.with_params(params), samples, values, bkd)

        def flat_loss(p: Array) -> Array:
            return bkd.reshape(loss(p), (1,))

        bkd.assert_allclose(
            loss.jacobian(params), bkd.jacobian(flat_loss, params), rtol=1e-10
        )

    def test_hvp_matches_autodiff(self, autodiff_bkd: AutodiffBackend[Array]) -> None:
        bkd = autodiff_bkd
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 15)))
        values = bkd.asarray(np.random.randn(1, 15))
        loss = SUPNMSELoss(supn.with_params(params), samples, values, bkd)
        direction = bkd.asarray(np.random.randn(supn.nparams()))

        def scalar_loss(p: Array) -> Array:
            return loss(p)[0, 0]

        bkd.assert_allclose(
            loss.hvp(params, direction)[:, 0],
            bkd.hvp(scalar_loss, params, direction),
            rtol=1e-10,
        )
