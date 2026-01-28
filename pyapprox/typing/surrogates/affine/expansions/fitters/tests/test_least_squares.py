"""Tests for LeastSquaresFitter.

Tests focus on fitter-specific behavior:
- Wrapping the solver correctly
- Returning correct result type
- Handling 1D values

Tests for basis_matrix() and with_params() are in
surrogates/affine/expansions/tests/test_expansions.py
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

from pyapprox.typing.surrogates.affine.expansions.fitters import (
    LeastSquaresFitter,
    DirectSolverResult,
)

from pyapprox.typing.surrogates.affine.univariate import create_bases_1d
from pyapprox.typing.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.typing.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.typing.surrogates.affine.expansions import BasisExpansion
from pyapprox.typing.probability import UniformMarginal


class TestLeastSquaresFitter(Generic[Array], unittest.TestCase):
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

        fitter = LeastSquaresFitter(self._bkd)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, DirectSolverResult)

    def test_result_params_shape(self) -> None:
        """Result params have correct shape."""
        expansion = self._create_expansion(nvars=2, max_level=3, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(2, 20))

        fitter = LeastSquaresFitter(self._bkd)
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.params().shape, (expansion.nterms(), 2))

    def test_handles_1d_values(self) -> None:
        """Fitter handles 1D values array."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        x, y = samples[0, :], samples[1, :]
        values_1d = x + y  # 1D array

        fitter = LeastSquaresFitter(self._bkd)
        result = fitter.fit(expansion, samples, values_1d)

        # Should work and produce correct result
        test_samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        x_test, y_test = test_samples[0, :], test_samples[1, :]
        expected = self._bkd.reshape(x_test + y_test, (1, -1))
        predicted = result(test_samples)

        self._bkd.assert_allclose(predicted, expected, rtol=1e-10, atol=1e-10)

    def test_matches_expansion_fit_method(self) -> None:
        """Fitter gives same result as expansion.fit()."""
        expansion_old = self._create_expansion(nvars=2, max_level=3)
        expansion_new = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        # Old way
        expansion_old.fit(samples, values)
        old_coef = expansion_old.get_coefficients()

        # New way
        fitter = LeastSquaresFitter(self._bkd)
        result = fitter.fit(expansion_new, samples, values)
        new_coef = result.params()

        self._bkd.assert_allclose(new_coef, old_coef, rtol=1e-10)


class TestLeastSquaresFitterNumpy(TestLeastSquaresFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestLeastSquaresFitterTorch(TestLeastSquaresFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
