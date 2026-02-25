"""Tests for BPDNFitter (Basis Pursuit Denoising).

Tests focus on:
- Returning SparseResult
- Producing sparse solutions
- Support/n_nonzero correctness
- Regularization behavior
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.bpdn import (
    BPDNFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    SparseResult,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestBPDNFitter(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_sparse_result(self) -> None:
        """Fit returns SparseResult."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, SparseResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        self.assertEqual(result.params().shape[1], 1)

    def test_produces_sparse_solution(self) -> None:
        """Higher penalty leads to sparser solution."""
        expansion_low = self._create_expansion(nvars=2, max_level=4)
        expansion_high = self._create_expansion(nvars=2, max_level=4)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        # Low regularization (less sparse)
        fitter_low = BPDNFitter(self._bkd, penalty=0.01)
        result_low = fitter_low.fit(expansion_low, samples, values)

        # High regularization (more sparse)
        fitter_high = BPDNFitter(self._bkd, penalty=0.5)
        result_high = fitter_high.fit(expansion_high, samples, values)

        # Higher penalty should give fewer nonzeros
        self.assertLessEqual(result_high.n_nonzero(), result_low.n_nonzero())

    def test_support_matches_nonzero(self) -> None:
        """Support contains exactly the nonzero coefficient indices."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = self._bkd.asarray(np.random.randn(1, 40))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # Count nonzeros manually
        params = result.params()[:, 0]
        tol = 1e-10
        abs_params = self._bkd.abs(params)
        nonzero_mask = abs_params > tol
        expected_support = self._bkd.where(nonzero_mask)[0]

        # support should match
        self._bkd.assert_allclose(result.support(), expected_support)

    def test_n_nonzero_correct(self) -> None:
        """n_nonzero matches actual count."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = self._bkd.asarray(np.random.randn(1, 40))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # n_nonzero should match support length
        self.assertEqual(result.n_nonzero(), len(result.support()))

    def test_invalid_penalty_raises(self) -> None:
        """Non-positive penalty raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            BPDNFitter(self._bkd, penalty=0.0)
        self.assertIn("positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            BPDNFitter(self._bkd, penalty=-1.0)
        self.assertIn("positive", str(ctx.exception))

    def test_multi_qoi_raises(self) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(2, 30))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_regularization_strength_stored(self) -> None:
        """Result stores regularization_strength correctly."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        penalty = 0.25
        fitter = BPDNFitter(self._bkd, penalty=penalty)
        result = fitter.fit(expansion, samples, values)

        self._bkd.assert_allclose(
            self._bkd.asarray([result.regularization_strength()]),
            self._bkd.asarray([penalty]),
        )

    def test_penalty_accessor(self) -> None:
        """Penalty accessor returns correct value."""
        fitter = BPDNFitter(self._bkd, penalty=0.75)
        self._bkd.assert_allclose(
            self._bkd.asarray([fitter.penalty()]),
            self._bkd.asarray([0.75]),
        )

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = BPDNFitter(self._bkd, penalty=0.1)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_recovers_target_expansion_with_decaying_coefficients(self) -> None:
        """BPDN can recover a sparse target expansion with decaying coefficients.

        Creates a 1D target expansion with 20 coefficients that decay as 1/k^2.
        Verifies that fitting with BPDN (small penalty) recovers the target with
        high accuracy.
        """
        bkd = self._bkd
        nvars = 1
        max_level = 19  # 20 terms for 1D

        # Create target expansion with decaying coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()

        # Create decaying coefficients: c_k = 1 / (k+1)^2
        # This gives coefficients like [1, 0.25, 0.11, 0.0625, ...]
        true_coef = self._bkd.asarray([[1.0 / ((k + 1) ** 2)] for k in range(nterms)])
        target_expansion = target_expansion.with_params(true_coef)

        # Generate training samples (more samples than coefficients)
        nsamples = 100
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Generate training values from target expansion
        values = target_expansion(samples)

        # Create expansion to fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)

        # Fit with small penalty (weak regularization) to allow recovery
        fitter = BPDNFitter(bkd, penalty=1e-6, max_iter=5000, tol=1e-8)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Fitted expansion should predict target with high accuracy
        # Use relative error since values can vary in magnitude
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        self.assertLess(float(rel_error), 1e-4, "Fitted expansion should match target")

        # Also verify coefficients are similar
        fitted_coef = result.params()
        coef_rel_error = bkd.norm(fitted_coef - true_coef) / bkd.norm(true_coef)
        self.assertLess(float(coef_rel_error), 1e-3, "Coefficients should be recovered")


class TestBPDNFitterNumpy(TestBPDNFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBPDNFitterTorch(TestBPDNFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
