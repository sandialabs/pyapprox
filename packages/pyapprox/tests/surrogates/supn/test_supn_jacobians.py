"""Tests for SUPN jacobians w.r.t. inputs and parameters.

Validates using DerivativeChecker and torch autograd.
"""

import numpy as np
import pytest
import torch

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.supn import create_supn
from pyapprox.util.backends.torch import TorchBkd


class TestSUPNInputJacobian:
    """Tests for SUPN jacobian w.r.t. inputs."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_jacobian_batch_shape(self, bkd) -> None:
        """Test jacobian_batch shape (nsamples, nqoi, nvars)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        jac = supn.jacobian_batch(samples)
        assert jac.shape == (10, 1, 2)

    def test_jacobian_batch_shape_multi_qoi(self, bkd) -> None:
        """Test jacobian_batch shape for nqoi > 1."""
        supn = create_supn(nvars=3, width=4, max_level=2, bkd=bkd, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, 7)))
        jac = supn.jacobian_batch(samples)
        assert jac.shape == (7, 2, 3)

    def test_jacobian_shape(self, bkd) -> None:
        """Test single-sample jacobian shape (nqoi, nvars)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        sample = bkd.asarray(np.random.uniform(-1, 1, (2, 1)))
        jac = supn.jacobian(sample)
        assert jac.shape == (1, 2)

    def test_jacobian_derivative_checker_1d(self, bkd) -> None:
        """Test 1D input jacobian via DerivativeChecker."""
        supn = create_supn(nvars=1, width=3, max_level=3, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=1,
            fun=lambda x: supn(x),
            jacobian=lambda x: supn.jacobian(x),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (1, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) < 1e-6

    def test_jacobian_derivative_checker_2d(self, bkd) -> None:
        """Test 2D input jacobian via DerivativeChecker."""
        supn = create_supn(nvars=2, width=4, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=2,
            fun=lambda x: supn(x),
            jacobian=lambda x: supn.jacobian(x),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) < 1e-6

    def test_jacobian_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test multi-QoI input jacobian via DerivativeChecker."""
        nqoi = 3
        nvars = 2
        supn = create_supn(nvars=nvars, width=4, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=nvars,
            fun=lambda x: supn(x),
            jacobian=lambda x: supn.jacobian(x),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (nvars, 1)))
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(
            sample, weights=weights, verbosity=0,
        )
        assert float(checker.error_ratio(errors[0])) < 1e-6


class TestSUPNParamJacobian:
    """Tests for SUPN jacobian w.r.t. parameters."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_jacobian_wrt_params_batch_shape(self, bkd) -> None:
        """Test shape (nsamples, nqoi, nparams)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        jac = supn.jacobian_wrt_params_batch(samples)
        assert jac.shape == (10, 1, supn.nparams())

    def test_jacobian_wrt_params_batch_shape_multi_qoi(self, bkd) -> None:
        """Test shape for nqoi > 1."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 7)))
        jac = supn.jacobian_wrt_params_batch(samples)
        assert jac.shape == (7, 2, supn.nparams())

    def test_jacobian_wrt_params_shape(self, bkd) -> None:
        """Test single-sample shape (nqoi, nparams)."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        sample = bkd.asarray(np.random.uniform(-1, 1, (2, 1)))
        jac = supn.jacobian_wrt_params(sample)
        assert jac.shape == (1, supn.nparams())

    def test_jacobian_wrt_params_derivative_checker(self, bkd) -> None:
        """Test param jacobian via DerivativeChecker for nqoi=1."""
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        # Wrap: input=params, output=supn(sample)
        function_obj = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=supn.nparams(),
            fun=lambda p: supn.with_params(p)(sample),
            jacobian=lambda p: supn.with_params(p).jacobian_wrt_params(sample),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True, verbosity=0,
        )
        assert float(checker.error_ratio(errors[0])) < 1e-6

    def test_jacobian_wrt_params_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test param jacobian via DerivativeChecker for nqoi > 1."""
        nqoi = 2
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        function_obj = FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=supn.nparams(),
            fun=lambda p: supn.with_params(p)(sample),
            jacobian=lambda p: supn.with_params(p).jacobian_wrt_params(sample),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(
            params[:, None], fd_eps=fd_eps, relative=True,
            weights=weights, verbosity=0,
        )
        assert float(checker.error_ratio(errors[0])) < 1e-6


class TestSUPNJacobianAutograd:
    """Verify analytical jacobians match torch autograd."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def test_input_jacobian_matches_autograd(self) -> None:
        """Verify input jacobian matches torch autograd."""
        bkd = self._bkd
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        analytical_jac = supn.jacobian(sample)  # (1, 2)

        # Autograd: compute per-output jacobian via backward
        sample_ag = sample.clone().requires_grad_(True)
        out = supn(sample_ag)  # (1, 1)
        out[0, 0].backward()
        autograd_jac = sample_ag.grad.T  # (1, 2)

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_param_jacobian_matches_autograd(self) -> None:
        """Verify param jacobian matches torch autograd."""
        bkd = self._bkd
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        analytical_jac = supn.jacobian_wrt_params(sample)  # (1, P)

        # Autograd: differentiate output w.r.t. flat params
        def f_from_params(p: torch.Tensor) -> torch.Tensor:
            return supn.with_params(p)(sample)  # (1, 1)

        # Compute row-by-row
        nqoi = supn.nqoi()
        P = supn.nparams()
        autograd_jac = torch.zeros(nqoi, P, dtype=torch.float64)
        for q in range(nqoi):
            p_tensor = params.clone().requires_grad_(True)
            out = supn.with_params(p_tensor)(sample)
            out[q, 0].backward()
            autograd_jac[q, :] = p_tensor.grad

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)

    def test_param_jacobian_autograd_multi_qoi(self) -> None:
        """Verify param jacobian matches autograd for nqoi > 1."""
        bkd = self._bkd
        nqoi = 2
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)

        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))
        analytical_jac = supn.jacobian_wrt_params(sample)  # (Q, P)

        P = supn.nparams()
        autograd_jac = torch.zeros(nqoi, P, dtype=torch.float64)
        for q in range(nqoi):
            p_tensor = params.clone().requires_grad_(True)
            out = supn.with_params(p_tensor)(sample)
            out[q, 0].backward()
            autograd_jac[q, :] = p_tensor.grad

        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-10)
