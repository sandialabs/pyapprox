"""Tests for FunctionTrain loss functions.

Tests validate FunctionTrainMSELoss gradient accuracy using DerivativeChecker
per CLAUDE.md convention.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal

from pyapprox.typing.surrogates.functiontrain import (
    FunctionTrain,
    create_additive_functiontrain,
    FunctionTrainMSELoss,
)
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestFunctionTrainMSELoss(Generic[Array], unittest.TestCase):
    """Tests for FunctionTrainMSELoss."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_univariate_expansion(
        self, max_level: int, nqoi: int = 1
    ) -> BasisExpansion[Array]:
        """Create a univariate polynomial expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(1, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def _create_additive_ft(
        self, nvars: int = 3, max_level: int = 2, nqoi: int = 1
    ) -> FunctionTrain[Array]:
        """Create an additive FunctionTrain for testing."""
        bkd = self._bkd
        univariate_bases = [
            self._create_univariate_expansion(max_level, nqoi)
            for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_initialization(self) -> None:
        """Test loss initialization."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Check nvars corresponds to number of parameters
        self.assertEqual(loss.nvars(), ft.nparams())
        self.assertEqual(loss.nqoi(), 1)

    def test_loss_evaluation(self) -> None:
        """Test basic loss evaluation."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        # Get current parameters
        params = ft._flatten_params()

        # Evaluate loss
        mse = loss(params)

        # Check output shape and properties
        self.assertEqual(mse.shape, (1, 1))
        self.assertTrue(bkd.all_bool(bkd.isfinite(mse)))

        # MSE should be non-negative
        self.assertGreaterEqual(float(mse[0, 0]), 0)

    def test_loss_changes_with_params(self) -> None:
        """Test that loss changes when parameters change."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

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
        self.assertNotEqual(float(mse1[0, 0]), float(mse2[0, 0]))

    def test_loss_zero_at_target(self) -> None:
        """Test that loss is zero when FT exactly matches target."""
        bkd = self._bkd
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=1)

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

    def test_jacobian_shape(self) -> None:
        """Test Jacobian output shape."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2, nqoi=1)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

        params = ft._flatten_params()
        grad = loss.jacobian(params)

        # Check gradient shape: (1, nparams)
        self.assertEqual(grad.shape, (1, loss.nvars()))
        self.assertTrue(bkd.all_bool(bkd.isfinite(grad)))

    def test_jacobian_zero_at_minimum(self) -> None:
        """Test that gradient is zero at the optimal solution."""
        bkd = self._bkd
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=1)

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

    def test_gradient_derivative_checker(self) -> None:
        """Test gradient using DerivativeChecker."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=1)

        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        loss = FunctionTrainMSELoss(ft, samples, values, bkd)

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
        self.assertTrue(
            bkd.all_bool(bkd.isfinite(grad_error)),
            "Gradient errors contain non-finite values",
        )

        # Error ratio should indicate good convergence
        error_ratio = float(checker.error_ratio(grad_error))
        self.assertLess(
            error_ratio, 1e-6, f"Error ratio {error_ratio:.2e} exceeds threshold"
        )

    def test_gradient_derivative_checker_multi_qoi(self) -> None:
        """Test gradient with multiple QoIs using DerivativeChecker."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        nqoi = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level, nqoi=nqoi)

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
        self.assertLess(error_ratio, 1e-6)


# NumPy backend tests
class TestFunctionTrainMSELossNumpy(TestFunctionTrainMSELoss[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestFunctionTrainMSELossTorch(TestFunctionTrainMSELoss[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

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

        bkd.assert_allclose(
            analytical_grad[0, :], autograd_grad, rtol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
