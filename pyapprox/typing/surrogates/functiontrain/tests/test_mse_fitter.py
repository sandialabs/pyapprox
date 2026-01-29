"""Tests for MSEFitter.

Tests validate the gradient-based MSE fitter for FunctionTrain surrogates.
Covers convergence, accuracy compared to ALS, and immutability.
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
    MSEFitter,
    ALSFitter,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


class TestMSEFitter(Generic[Array], unittest.TestCase):
    """Tests for MSEFitter."""

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

    def test_mse_rejects_non_functiontrain(self) -> None:
        """Test that MSEFitter raises TypeError for non-FunctionTrain."""
        bkd = self._bkd
        fitter = MSEFitter(bkd)

        # Try to fit a BasisExpansion instead of FunctionTrain
        expansion = self._create_univariate_expansion(2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with self.assertRaises(TypeError) as ctx:
            fitter.fit(expansion, samples, values)

        self.assertIn("MSEFitter only works with FunctionTrain", str(ctx.exception))

    def test_mse_fits_own_output(self) -> None:
        """Test MSE can fit FunctionTrain to its own output (exact recovery)."""
        bkd = self._bkd
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Set known coefficients
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams))
        ft_with_true_params = ft.with_params(true_params)

        # Generate training data from the FT itself
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_with_true_params(samples)

        # Randomize parameters before fitting
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_randomized = ft.with_params(random_params)

        # Fit with MSE
        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_randomized, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-5, atol=1e-5)

    def test_mse_fits_additive_polynomial(self) -> None:
        """Test MSE fits additive polynomial functions accurately."""
        bkd = self._bkd
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Set specific coefficients
        nparams = ft.nparams()
        params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(params)

        # Generate training data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Start from random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_init, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-5, atol=1e-5)

    def test_mse_immutability(self) -> None:
        """Test that MSE doesn't modify original FunctionTrain."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=3, max_level=1)

        # Get original parameters
        original_params = bkd.copy(ft._flatten_params())

        # Fit with some data
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = MSEFitter(bkd)
        _ = fitter.fit(ft, samples, values)

        # Original should be unchanged
        bkd.assert_allclose(ft._flatten_params(), original_params, rtol=1e-12)

    def test_mse_result_attributes(self) -> None:
        """Test that MSEFitterResult contains expected attributes."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft, samples, values)

        # Check result attributes
        self.assertIsInstance(result.surrogate(), FunctionTrain)
        self.assertIsInstance(result.final_loss(), float)
        self.assertIsInstance(result.converged(), bool)
        # n_iterations is optimizer-specific, access via raw result
        raw_result = result.optimizer_result().get_raw_result()
        self.assertTrue(hasattr(raw_result, "nit"))

    def test_mse_validates_dimension_mismatch(self) -> None:
        """Test that MSE validates input dimensions."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=3, max_level=1)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))  # Wrong nvars
        values = bkd.asarray(np.random.randn(1, 10))

        fitter = MSEFitter(bkd)

        with self.assertRaises(ValueError):
            fitter.fit(ft, samples, values)

    def test_mse_handles_1d_values(self) -> None:
        """Test that MSE handles 1D values array."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values_1d = bkd.asarray(np.random.randn(nsamples))  # 1D

        fitter = MSEFitter(bkd)
        result = fitter.fit(ft, samples, values_1d)

        # Should work without error
        self.assertIsInstance(result.surrogate(), FunctionTrain)

    def test_mse_custom_optimizer(self) -> None:
        """Test MSE with custom optimizer settings."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        # Configure custom optimizer
        optimizer = ScipyTrustConstrOptimizer(
            verbosity=0, maxiter=100, gtol=1e-6
        )

        fitter = MSEFitter(bkd)
        fitter.set_optimizer(optimizer)

        # Verify optimizer is set
        self.assertIsNotNone(fitter.optimizer())

        result = fitter.fit(ft, samples, values)
        self.assertIsInstance(result.surrogate(), FunctionTrain)

    def test_mse_loss_decreases(self) -> None:
        """Test that final loss is less than initial loss."""
        bkd = self._bkd
        nvars = 2
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Generate target from random params
        nparams = ft.nparams()
        target_params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(target_params)
        values = ft_target(samples)

        # Start far from solution
        init_params = bkd.asarray(np.random.randn(nparams) * 5.0)
        ft_init = ft.with_params(init_params)

        # Compute initial loss
        from pyapprox.typing.surrogates.functiontrain import FunctionTrainMSELoss

        loss = FunctionTrainMSELoss(ft_init, samples, values, bkd)
        initial_loss = float(loss(init_params)[0, 0])

        # Fit
        fitter = MSEFitter(bkd)
        result = fitter.fit(ft_init, samples, values)

        # Final loss should be less than initial
        self.assertLess(result.final_loss(), initial_loss)

    def test_mse_comparable_to_als(self) -> None:
        """Test that MSE achieves similar accuracy to ALS on simple problems."""
        bkd = self._bkd
        nvars = 3
        max_level = 1
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Set known parameters
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams) * 2.0)
        ft_target = ft.with_params(true_params)

        # Generate training data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        # Fit with ALS
        als_fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-10)
        als_result = als_fitter.fit(ft_init, samples, values)
        als_predictions = als_result.surrogate()(samples)

        # Fit with MSE
        mse_fitter = MSEFitter(bkd)
        mse_result = mse_fitter.fit(ft_init, samples, values)
        mse_predictions = mse_result.surrogate()(samples)

        # Both should achieve good fit
        bkd.assert_allclose(als_predictions, values, rtol=1e-4, atol=1e-4)
        bkd.assert_allclose(mse_predictions, values, rtol=1e-4, atol=1e-4)


# NumPy backend tests
class TestMSEFitterNumpy(TestMSEFitter[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestMSEFitterTorch(TestMSEFitter[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
