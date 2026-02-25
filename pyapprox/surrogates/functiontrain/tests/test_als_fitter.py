"""Tests for ALSFitter."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    ALSFitter,
    FunctionTrain,
    create_additive_functiontrain,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestALSFitter(Generic[Array], unittest.TestCase):
    """Base class for ALSFitter tests."""

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
            self._create_univariate_expansion(max_level, nqoi) for _ in range(nvars)
        ]
        return create_additive_functiontrain(univariate_bases, bkd, nqoi)

    def test_als_rejects_non_functiontrain(self) -> None:
        """Test that ALSFitter raises TypeError for non-FunctionTrain."""
        bkd = self._bkd
        fitter = ALSFitter(bkd)

        # Try to fit a BasisExpansion instead of FunctionTrain
        expansion = self._create_univariate_expansion(2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        values = bkd.asarray(np.random.randn(1, 10))

        with self.assertRaises(TypeError) as ctx:
            fitter.fit(expansion, samples, values)

        self.assertIn("ALSFitter only works with FunctionTrain", str(ctx.exception))

    def test_als_fits_own_output(self) -> None:
        """Test ALS can fit FunctionTrain to its own output (exact recovery).

        Following the legacy test pattern: set coefficients, evaluate,
        then fit to that output. This ensures exact recovery is possible.
        """
        bkd = self._bkd
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Set known coefficients on all trainable expansions
        nparams = ft.nparams()
        true_params = bkd.asarray(np.ones(nparams))
        ft_with_true_params = ft.with_params(true_params)

        # Generate training data from the FT itself
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_with_true_params(samples)

        # Randomize parameters before fitting
        random_params = bkd.asarray(np.random.randn(nparams))
        ft_randomized = ft.with_params(random_params)

        # Fit with ALS - should recover exact fit
        fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-10)
        result = fitter.fit(ft_randomized, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_fits_additive_polynomial(self) -> None:
        """Test ALS fits additive polynomial functions accurately.

        Uses polynomials that are exactly representable by the model.
        """
        bkd = self._bkd
        nvars = 3
        max_level = 2  # Quadratic polynomials
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Set specific coefficients to create a known additive function
        # Each univariate expansion has (max_level + 1) = 3 terms
        nparams = ft.nparams()
        # Create coefficients that give a non-trivial additive function
        params = bkd.asarray(np.random.randn(nparams))
        ft_target = ft.with_params(params)

        # Generate training data from this target
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Start from random initialization
        random_params = bkd.asarray(np.random.randn(nparams) * 0.1)
        ft_init = ft.with_params(random_params)

        fitter = ALSFitter(bkd, max_sweeps=20, tol=1e-12)
        result = fitter.fit(ft_init, samples, values)

        # Check that fitted model predicts correctly
        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_converges_for_additive(self) -> None:
        """Test that ALS converges for additive FT.

        For additive FunctionTrain, the tensor structure allows rapid
        convergence.
        """
        bkd = self._bkd
        nvars = 3
        max_level = 2
        ft = self._create_additive_ft(nvars=nvars, max_level=max_level)

        # Create target with known params
        nparams = ft.nparams()
        target_params = bkd.asarray(np.ones(nparams) * 2.0)
        ft_target = ft.with_params(target_params)

        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = ft_target(samples)

        # Random initialization
        random_params = bkd.asarray(np.random.randn(nparams))
        ft_init = ft.with_params(random_params)

        # ALS should converge quickly for additive structure
        fitter = ALSFitter(bkd, max_sweeps=5, tol=1e-12)
        result = fitter.fit(ft_init, samples, values)

        fitted_ft = result.surrogate()
        predictions = fitted_ft(samples)

        # Should get very good fit
        bkd.assert_allclose(predictions, values, rtol=1e-6, atol=1e-6)

    def test_als_immutability(self) -> None:
        """Test that ALS doesn't modify original FunctionTrain."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=3, max_level=1)

        # Get original parameters
        original_params = bkd.copy(ft._flatten_params())

        # Fit with some data
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (3, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = ALSFitter(bkd, max_sweeps=5)
        _ = fitter.fit(ft, samples, values)

        # Original should be unchanged
        bkd.assert_allclose(ft._flatten_params(), original_params, rtol=1e-12)

    def test_als_result_contains_surrogate(self) -> None:
        """Test that ALSFitterResult contains the fitted surrogate."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = bkd.asarray(np.random.randn(1, nsamples))

        fitter = ALSFitter(bkd, max_sweeps=3)
        result = fitter.fit(ft, samples, values)

        # Check result attributes
        self.assertIsInstance(result.surrogate(), FunctionTrain)
        self.assertIsInstance(result.n_sweeps(), int)
        self.assertIsInstance(result.residual_history(), list)
        self.assertIsInstance(result.converged(), bool)

    def test_als_residual_decreases(self) -> None:
        """Test that residual generally decreases over sweeps."""
        bkd = self._bkd
        nvars = 3
        ft = self._create_additive_ft(nvars=nvars, max_level=2)

        # Generate noisy data
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        true_values = samples[0:1, :] + samples[1:2, :] ** 2
        noise = bkd.asarray(np.random.randn(1, nsamples) * 0.01)
        values = true_values + noise

        fitter = ALSFitter(bkd, max_sweeps=10, tol=1e-12)
        result = fitter.fit(ft, samples, values)

        history = result.residual_history()

        # Initial residual should be larger than final
        self.assertGreater(history[0], history[-1])

    def test_als_handles_1d_values(self) -> None:
        """Test that ALS handles 1D values array."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=2, max_level=1)

        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values_1d = bkd.asarray(np.random.randn(nsamples))  # 1D

        fitter = ALSFitter(bkd, max_sweeps=3)
        result = fitter.fit(ft, samples, values_1d)

        # Should work without error
        self.assertIsInstance(result.surrogate(), FunctionTrain)

    def test_als_validates_dimension_mismatch(self) -> None:
        """Test that ALS validates input dimensions."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=3, max_level=1)

        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))  # Wrong nvars
        values = bkd.asarray(np.random.randn(1, 10))

        fitter = ALSFitter(bkd)

        with self.assertRaises(ValueError):
            fitter.fit(ft, samples, values)

    def test_als_convergence_flag(self) -> None:
        """Test that convergence flag is set correctly."""
        bkd = self._bkd
        ft = self._create_additive_ft(nvars=2, max_level=1)

        # Simple linear function that should converge quickly
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        values = samples[0:1, :]  # Just x_0

        # With very tight tolerance and few sweeps - may not converge
        fitter_strict = ALSFitter(bkd, max_sweeps=1, tol=1e-20)
        fitter_strict.fit(ft, samples, values)

        # With loose tolerance - should converge
        fitter_loose = ALSFitter(bkd, max_sweeps=10, tol=1.0)
        result_loose = fitter_loose.fit(ft, samples, values)

        # Loose tolerance should converge
        self.assertTrue(result_loose.converged())


# NumPy backend tests
class TestALSFitterNumpy(TestALSFitter[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestALSFitterTorch(TestALSFitter[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
