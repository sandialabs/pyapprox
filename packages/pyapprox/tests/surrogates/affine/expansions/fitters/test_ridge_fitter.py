"""Tests for RidgeFitter.

Tests focus on:
- Wrapping the solver correctly
- Returning correct result type
- Regularization behavior
- Invalid alpha validation
"""

import numpy as np
import pytest

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


class TestRidgeFitter:
    """Base test class."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_direct_solver_result(self, bkd) -> None:
        """Fit returns DirectSolverResult."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        fitter = RidgeFitter(bkd, alpha=1.0)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(2, 20))

        fitter = RidgeFitter(bkd, alpha=0.1)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 2)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values_1d = bkd.asarray(np.random.randn(50))

        fitter = RidgeFitter(bkd, alpha=0.01)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work without error
        assert result.params().shape[1] == 1

    def test_regularization_reduces_norm(self, bkd) -> None:
        """Higher alpha leads to smaller coefficient norm."""
        expansion_low = self._create_expansion(bkd, nvars=2, max_level=3)
        expansion_high = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        # Low regularization
        fitter_low = RidgeFitter(bkd, alpha=0.01)
        result_low = fitter_low.fit(expansion_low, samples, values)
        norm_low = float(bkd.norm(result_low.params()))

        # High regularization
        fitter_high = RidgeFitter(bkd, alpha=10.0)
        result_high = fitter_high.fit(expansion_high, samples, values)
        norm_high = float(bkd.norm(result_high.params()))

        # Higher alpha should give smaller norm
        assert norm_high < norm_low

    def test_small_alpha_matches_lstsq(self, bkd) -> None:
        """With very small alpha, result approaches least squares solution."""
        expansion_ridge = self._create_expansion(bkd, nvars=2, max_level=3)
        expansion_lstsq = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        # Very small alpha (essentially unregularized)
        fitter_ridge = RidgeFitter(bkd, alpha=1e-10)
        result_ridge = fitter_ridge.fit(expansion_ridge, samples, values)

        # True least squares
        fitter_lstsq = LeastSquaresFitter(bkd)
        result_lstsq = fitter_lstsq.fit(expansion_lstsq, samples, values)

        # Should be very close
        bkd.assert_allclose(
            result_ridge.params(), result_lstsq.params(), rtol=1e-5
        )

    def test_invalid_alpha_raises(self, bkd) -> None:
        """Non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            RidgeFitter(bkd, alpha=0.0)

        with pytest.raises(ValueError, match="positive"):
            RidgeFitter(bkd, alpha=-1.0)

    def test_alpha_accessor(self, bkd) -> None:
        """Alpha accessor returns correct value."""
        fitter = RidgeFitter(bkd, alpha=2.5)
        bkd.assert_allclose(
            bkd.asarray([fitter.alpha()]),
            bkd.asarray([2.5]),
        )

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Fitted surrogate can evaluate at new points."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(1, 30))

        fitter = RidgeFitter(bkd, alpha=0.1)
        result = fitter.fit(expansion, samples, values)

        # Evaluate at new points
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))
        predictions = result(test_samples)

        assert predictions.shape == (1, 10)
