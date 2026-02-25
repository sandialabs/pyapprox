"""Tests for OMPFitter (Orthogonal Matching Pursuit).

Tests focus on:
- Returning OMPResult
- Respecting max_nonzeros limit
- Respecting rtol stopping criterion
- Selection order and residual history tracking
- Sparse signal recovery
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.expansions.fitters.omp import (
    OMPFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    OMPResult,
)
from pyapprox.optimization.linear.sparse import OMPTerminationFlag

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestOMPFitter(Generic[Array], unittest.TestCase):
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

    def test_fit_returns_omp_result(self) -> None:
        """Fit returns OMPResult."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, OMPResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 1))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        self.assertEqual(result.params().shape[1], 1)

    def test_respects_max_nonzeros(self) -> None:
        """Selects at most max_nonzeros terms."""
        expansion = self._create_expansion(nvars=2, max_level=5)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        max_nonzeros = 3
        fitter = OMPFitter(self._bkd, max_nonzeros=max_nonzeros, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        self.assertLessEqual(result.n_nonzero(), max_nonzeros)
        self.assertLessEqual(len(result.selection_order()), max_nonzeros)

    def test_respects_rtol(self) -> None:
        """Stops when residual norm below rtol."""
        bkd = self._bkd
        nvars = 1
        max_level = 10

        # Create a target with only 2 nonzero coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        true_coef[0, 0] = 1.0
        true_coef[1, 0] = 0.5
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit with high max_nonzeros but tight rtol
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = OMPFitter(bkd, max_nonzeros=nterms, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Should stop early due to rtol, not max_nonzeros
        self.assertEqual(
            result.termination_flag(), OMPTerminationFlag.RESIDUAL_TOLERANCE
        )
        # Should select ~2 terms (the true sparse support)
        self.assertLessEqual(result.n_nonzero(), 4)

    def test_selection_order_recorded(self) -> None:
        """selection_order has correct length and values."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = self._bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(self._bkd, max_nonzeros=5, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        # selection_order should have n_nonzero entries
        self.assertEqual(len(result.selection_order()), result.n_nonzero())

        # selection_order should match support (for OMP they're the same)
        self._bkd.assert_allclose(result.selection_order(), result.support())

    def test_residual_history_recorded(self) -> None:
        """residual_history has correct length."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = self._bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(self._bkd, max_nonzeros=5, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        # residual_history should have n_nonzero entries (one per iteration)
        self.assertEqual(len(result.residual_history()), result.n_nonzero())

    def test_residual_history_decreasing(self) -> None:
        """Residual norm decreases (or stays same) each iteration."""
        expansion = self._create_expansion(nvars=2, max_level=4)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        fitter = OMPFitter(self._bkd, max_nonzeros=8, rtol=1e-15)
        result = fitter.fit(expansion, samples, values)

        residuals = result.residual_history()
        for ii in range(len(residuals) - 1):
            # Each residual should be <= previous (OMP is greedy optimal)
            self.assertLessEqual(
                float(residuals[ii + 1]),
                float(residuals[ii]) + 1e-10,  # small tolerance for numerics
            )

    def test_n_nonzero_correct(self) -> None:
        """n_nonzero matches support length."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 40)))
        values = self._bkd.asarray(np.random.randn(1, 40))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.n_nonzero(), len(result.support()))

    def test_invalid_max_nonzeros_raises(self) -> None:
        """max_nonzeros < 1 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OMPFitter(self._bkd, max_nonzeros=0)
        self.assertIn("max_nonzeros", str(ctx.exception))

    def test_multi_qoi_raises(self) -> None:
        """nqoi > 1 raises ValueError."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(2, 30))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        with self.assertRaises(ValueError) as ctx:
            fitter.fit(expansion, samples, values)
        self.assertIn("nqoi=1", str(ctx.exception))

    def test_accessors(self) -> None:
        """Accessors return correct values."""
        fitter = OMPFitter(self._bkd, max_nonzeros=7, rtol=0.01)
        self.assertEqual(fitter.max_nonzeros(), 7)
        self._bkd.assert_allclose(
            self._bkd.asarray([fitter.rtol()]),
            self._bkd.asarray([0.01]),
        )

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = OMPFitter(self._bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))

    def test_recovers_sparse_signal(self) -> None:
        """OMP recovers a known sparse signal accurately."""
        bkd = self._bkd
        nvars = 1
        max_level = 19  # 20 terms

        # Create target with exactly 3 nonzero coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()
        true_coef = bkd.zeros((nterms, 1))
        # Set 3 nonzero coefficients at indices 0, 2, 5
        true_coef[0, 0] = 1.0
        true_coef[2, 0] = -0.5
        true_coef[5, 0] = 0.3
        target_expansion = target_expansion.with_params(true_coef)

        # Generate noiseless data
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        values = target_expansion(samples)

        # Fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        fitter = OMPFitter(bkd, max_nonzeros=5, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Should recover the 3 terms
        self.assertLessEqual(result.n_nonzero(), 5)

        # Evaluate at test points
        ntest = 50
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Should predict accurately
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        self.assertLess(float(rel_error), 1e-8)

    def test_recovers_target_expansion_with_decaying_coefficients(self) -> None:
        """OMP can recover a target expansion with decaying coefficients.

        Creates a 1D target expansion with 20 coefficients that decay as 1/k^2.
        Verifies that fitting with OMP recovers the target with high accuracy.
        """
        bkd = self._bkd
        nvars = 1
        max_level = 19  # 20 terms for 1D

        # Create target expansion with decaying coefficients
        target_expansion = self._create_expansion(nvars=nvars, max_level=max_level)
        nterms = target_expansion.nterms()

        # Create decaying coefficients: c_k = 1 / (k+1)^2
        # This gives coefficients like [1, 0.25, 0.11, 0.0625, ...]
        true_coef = self._bkd.asarray(
            [[1.0 / ((k + 1) ** 2)] for k in range(nterms)]
        )
        target_expansion = target_expansion.with_params(true_coef)

        # Generate training samples (more samples than coefficients)
        nsamples = 100
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Generate training values from target expansion
        values = target_expansion(samples)

        # Create expansion to fit
        fit_expansion = self._create_expansion(nvars=nvars, max_level=max_level)

        # Fit with OMP - allow enough terms and tight tolerance
        fitter = OMPFitter(bkd, max_nonzeros=nterms, rtol=1e-10)
        result = fitter.fit(fit_expansion, samples, values)

        # Evaluate at test points
        ntest = 50
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (nvars, ntest)))
        target_values = target_expansion(test_samples)
        fitted_values = result(test_samples)

        # Fitted expansion should predict target with high accuracy
        rel_error = bkd.norm(fitted_values - target_values) / bkd.norm(target_values)
        self.assertLess(float(rel_error), 1e-4, "Fitted expansion should match target")

        # Also verify coefficients are similar
        fitted_coef = result.params()
        coef_rel_error = bkd.norm(fitted_coef - true_coef) / bkd.norm(true_coef)
        self.assertLess(float(coef_rel_error), 1e-3, "Coefficients should be recovered")


class TestOMPFitterNumpy(TestOMPFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOMPFitterTorch(TestOMPFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
