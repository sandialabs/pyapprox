"""Tests for FunctionTrain loss functions.

Tests validate FunctionTrainMSELoss gradient accuracy using DerivativeChecker
per CLAUDE.md convention.
"""

import numpy as np
import pytest
import torch

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    FunctionTrainMSELoss,
    create_additive_functiontrain,
)
from pyapprox.util.backends.torch import TorchBkd


class TestFunctionTrainMSELoss:
    """Tests for FunctionTrainMSELoss."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_univariate_expansion(self, bkd, max_level, nqoi=1):
        """Create a univariate polynomial expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, bkd, nvars=3, max_level=2, nqoi=1):
        """Create an additive FunctionTrain for testing."""
        univariate_bases = [
            self._create_univariate_expansion(bkd, max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_initialization(self, bkd) -> None:
        """Test loss initialization."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Check nvars corresponds to number of parameters
        assert loss.nvars() == ft.nparams()
        assert loss.nqoi() == 1

    def test_loss_evaluation(self, bkd) -> None:
        """Test basic loss evaluation."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Get current parameters
        params = ft._flatten_params()

        # Evaluate loss
        mse = loss(params)

        # Check output shape and properties
        assert mse.shape == (1, 1)
        assert bkd.all_bool(bkd.isfinite(mse))

        # MSE should be non-negative
        assert float(mse[0, 0]) >= 0

    def test_loss_changes_with_params(self, bkd) -> None:
        """Test that loss changes when parameters change."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Get current parameters
        params1 = ft._flatten_params()
        mse1 = loss(params1)

        # Perturb parameters
        params2 = params1 + bkd.asarray(np.full(params1.shape, 0.5))
        mse2 = loss(params2)

        # Loss should be different
        assert float(mse1[0, 0]) != float(mse2[0, 0])

    def test_loss_zero_at_target(self, bkd) -> None:
        """Test that loss is zero when FT exactly matches target."""
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=1)

        # Set known parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Generate values from FT itself
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft(samples)

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Loss at true parameters should be zero
        mse = loss(params)
        bkd.assert_allclose(mse, bkd.zeros((1, 1)), atol=1e-12)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian output shape."""
        nvars = 3
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        params = ft._flatten_params()
        grad = loss.jacobian(params)

        # Check gradient shape: (1, nparams)
        assert grad.shape == (1, loss.nvars())
        assert bkd.all_bool(bkd.isfinite(grad))

    def test_jacobian_zero_at_minimum(self, bkd) -> None:
        """Test that gradient is zero at the optimal solution."""
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=1)

        # Set known parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft = ft.with_params(params)

        # Generate values from FT itself
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft(samples)

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Gradient at true parameters should be zero
        grad = loss.jacobian(params)
        bkd.assert_allclose(grad, bkd.zeros((1, nparams)), atol=1e-10)

    def test_gradient_derivative_checker(self, bkd) -> None:
        """Test gradient using DerivativeChecker."""
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=1)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        # Set non-zero parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft_with_params = ft.with_params(params)

        # Create a new loss with updated FT
        loss = FunctionTrainMSELoss(ft_with_params, samples, values, bkd)

        # Use DerivativeChecker
        checker = DerivativeChecker(loss)

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nparams, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        # Get gradient error
        grad_error = errors[0]

        # All errors should be finite
        assert bkd.all_bool(bkd.isfinite(grad_error)), (
            "Gradient errors contain non-finite values"
        )

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        assert error_ratio < 1e-6, f"Error ratio {error_ratio:.2e} exceeds threshold"

    def test_gradient_derivative_checker_multi_qoi(self, bkd) -> None:
        """Test gradient with multiple QoIs using DerivativeChecker."""
        nvars = 2
        max_level = 1
        nqoi = 2
        ft = self._create_additive_ft(bkd, nvars=nvars, max_level=max_level, nqoi=nqoi)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(nqoi, nsamples))

        # Set non-zero parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft_with_params = ft.with_params(params)

        loss = FunctionTrainMSELoss(ft_with_params, samples, values, bkd)

        checker = DerivativeChecker(loss)
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        errors = checker.check_derivatives(
            params[:, None],
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        error_ratio = float(checker.error_ratio(errors[0]))
        assert error_ratio < 1e-6


# Torch-only test
class TestFunctionTrainMSELossAutograd:
    """Verify analytical gradient matches torch autograd."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        np.random.seed(42)

    def _create_univariate_expansion(self, max_level, nqoi=1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(self, nvars=3, max_level=2, nqoi=1):
        bkd = self._bkd
        univariate_bases = [
            self._create_univariate_expansion(max_level, nqoi) for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_gradient_matches_autograd(self) -> None:
        """Verify analytical gradient matches torch autograd."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        nqoi = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(nqoi, nsamples))

        # Set non-zero parameters
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft_with_params = ft.with_params(params)

        loss = FunctionTrainMSELoss(ft_with_params, samples, values, bkd)

        # Get analytical gradient
        analytical_grad = loss.jacobian(params)  # (1, nparams)

        # Get autograd gradient
        def loss_from_params(p: torch.Tensor) -> torch.Tensor:
            return loss(p)[0, 0]

        autograd_grad = torch.autograd.functional.jacobian(loss_from_params, params)
        # Shape: (nparams,)

        bkd.assert_allclose(analytical_grad[0, :], autograd_grad, rtol=1e-10)
