"""Tests for LeastSquaresFitter.

Tests focus on fitter-specific behavior:
- Wrapping the solver correctly
- Returning correct result type
- Handling 1D values

Tests for basis_matrix() and with_params() are in
surrogates/affine/expansions/tests/test_expansions.py
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters import (
    DirectSolverResult,
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d


class TestLeastSquaresFitter:
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

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, DirectSolverResult)

    def test_result_params_shape(self, bkd) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(2, 20))

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion, samples, values)

        assert result.params().shape == (expansion.nterms(), 2)

    def test_handles_1d_values(self, bkd) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        x, y = samples[0, :], samples[1, :]
        values_1d = x + y  # 1D array

        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work and produce correct result
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = bkd.reshape(x_test + y_test, (1, -1))
        predicted = result(test_samples)

        bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_matches_expansion_fit_method(self, bkd) -> None:
        """Fitter gives same result as expansion.fit()."""
        expansion_old = self._create_expansion(bkd, nvars=2, max_level=3)
        expansion_new = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        # Old way
        expansion_old.fit(samples, values)
        old_coef = expansion_old.get_coefficients()

        # New way
        fitter = LeastSquaresFitter(bkd)
        result = fitter.fit(expansion_new, samples, values)
        new_coef = result.params()

        bkd.assert_allclose(new_coef, old_coef, rtol=1e-10)
