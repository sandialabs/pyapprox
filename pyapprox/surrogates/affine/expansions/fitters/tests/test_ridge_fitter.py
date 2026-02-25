"""Tests for RidgeFitter.

Tests focus on:
- Wrapping the solver correctly
- Returning correct result type
- Regularization behavior
- Invalid alpha validation
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.least_squares import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    DirectSolverResult,
)
from pyapprox.surrogates.affine.expansions.fitters.ridge import (
    RidgeFitter,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestRidgeFitter(Generic[Array], unittest.TestCase):
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

    def test_fit_returns_direct_solver_result(self) -> None:
        """Fit returns DirectSolverResult."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        fitter = RidgeFitter(self._bkd, alpha=1.0)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(2, 20))

        fitter = RidgeFitter(self._bkd, alpha=0.1)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 2))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = self._bkd.asarray(np.random.randn(50))

        fitter = RidgeFitter(self._bkd, alpha=0.01)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        self.assertEqual(result.params().shape[1], 1)

    def test_regularization_reduces_norm(self) -> None:
        """Higher alpha leads to smaller coefficient norm."""
        expansion_low = self._create_expansion(nvars=2, max_level=3)
        expansion_high = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        # Low regularization
        fitter_low = RidgeFitter(self._bkd, alpha=0.01)
        result_low = fitter_low.fit(expansion_low, samples, values)
        norm_low = float(self._bkd.norm(result_low.params()))

        # High regularization
        fitter_high = RidgeFitter(self._bkd, alpha=10.0)
        result_high = fitter_high.fit(expansion_high, samples, values)
        norm_high = float(self._bkd.norm(result_high.params()))

        # Higher alpha should give smaller norm
        self.assertLess(norm_high, norm_low)

    def test_small_alpha_matches_lstsq(self) -> None:
        """With very small alpha, result approaches least squares solution."""
        expansion_ridge = self._create_expansion(nvars=2, max_level=3)
        expansion_lstsq = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        # Very small alpha (essentially unregularized)
        fitter_ridge = RidgeFitter(self._bkd, alpha=1e-10)
        result_ridge = fitter_ridge.fit(expansion_ridge, samples, values)

        # True least squares
        fitter_lstsq = LeastSquaresFitter(self._bkd)
        result_lstsq = fitter_lstsq.fit(expansion_lstsq, samples, values)

        # Should be very close
        self._bkd.assert_allclose(
            result_ridge.params(), result_lstsq.params(), rtol=1e-5
        )

    def test_invalid_alpha_raises(self) -> None:
        """Non-positive alpha raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            RidgeFitter(self._bkd, alpha=0.0)
        self.assertIn("positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            RidgeFitter(self._bkd, alpha=-1.0)
        self.assertIn("positive", str(ctx.exception))

    def test_alpha_accessor(self) -> None:
        """Alpha accessor returns correct value."""
        fitter = RidgeFitter(self._bkd, alpha=2.5)
        self._bkd.assert_allclose(
            self._bkd.asarray([fitter.alpha()]),
            self._bkd.asarray([2.5]),
        )

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(1, 30))

        fitter = RidgeFitter(self._bkd, alpha=0.1)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        self.assertEqual(predictions.shape, (1, 10))


class TestRidgeFitterNumpy(TestRidgeFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestRidgeFitterTorch(TestRidgeFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
