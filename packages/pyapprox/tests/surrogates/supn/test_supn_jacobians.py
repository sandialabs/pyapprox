"""Tests for SUPN jacobians w.r.t. inputs and parameters.

Validates using DerivativeChecker and the backend's autodiff Jacobian.
"""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.supn import create_supn
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array


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
            nqoi=1,
            nvars=1,
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
            nqoi=1,
            nvars=2,
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
            nqoi=nqoi,
            nvars=nvars,
            fun=lambda x: supn(x),
            jacobian=lambda x: supn.jacobian(x),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (nvars, 1)))
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(
            sample,
            weights=weights,
            verbosity=0,
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
            nqoi=1,
            nvars=supn.nparams(),
            fun=lambda p: supn.with_params(p)(sample),
            jacobian=lambda p: supn.with_params(p).jacobian_wrt_params(sample),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
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
            nqoi=nqoi,
            nvars=supn.nparams(),
            fun=lambda p: supn.with_params(p)(sample),
            jacobian=lambda p: supn.with_params(p).jacobian_wrt_params(sample),
            bkd=bkd,
        )
        checker = DerivativeChecker(function_obj)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))
        weights = bkd.asarray(np.random.randn(nqoi, 1))
        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            weights=weights,
            verbosity=0,
        )
        assert float(checker.error_ratio(errors[0])) < 1e-6


class TestSUPNParamJacobianAutodiff:
    """The analytical parameter Jacobian matches the backend's autodiff
    Jacobian, which needs the computation graph to survive. The input
    Jacobian's autodiff check is in test_supn_jacobians_torch.py."""

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        np.random.seed(42)

    @pytest.mark.parametrize("nqoi", [1, 2])
    def test_param_jacobian_matches_autodiff(
        self, torch_bkd: AutodiffBackend[Array], nqoi: int
    ) -> None:
        bkd = torch_bkd
        supn = create_supn(nvars=2, width=3, max_level=2, bkd=bkd, nqoi=nqoi)
        params = bkd.asarray(np.random.randn(supn.nparams()))
        supn = supn.with_params(params)
        sample = bkd.asarray(np.random.uniform(-0.9, 0.9, (2, 1)))

        def flat_output(p: Array) -> Array:
            return bkd.reshape(supn.with_params(p)(sample), (nqoi,))

        bkd.assert_allclose(
            supn.jacobian_wrt_params(sample),  # (nqoi, nparams)
            bkd.jacobian(flat_output, params),
            rtol=1e-10,
        )
