"""Tests for PCEDegreeSelectionFitter.

Tests verify that cross-validation-based level selection correctly
identifies the optimal index set level for various test functions.
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

from pyapprox.surrogates.affine.expansions.fitters.pce_cv import (
    PCEDegreeSelectionFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
    HyperbolicIndexSequence,
    SparseGridIndexSequence,
)
from pyapprox.surrogates.affine.indices.growth_rules import (
    LinearGrowthRule,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestPCEDegreeSelectionFitter(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion with initial indices."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_selects_correct_degree_for_quadratic(self) -> None:
        """For x^2, selects level >= 2 and CV score at level 2 is near zero."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # level 1 should have large error, levels >= 2 should be near zero
        cv_scores = result.cv_scores()
        bkd.assert_allclose(
            cv_scores[1:2], bkd.zeros(1), atol=1e-12
        )
        # level 1 CV score should be much larger
        self.assertGreater(float(cv_scores[0]), 1e-2)
        # selected level should be >= 2 (exact choice among >=2 is noise)
        self.assertGreaterEqual(result.best_label(), 2)

    def test_selects_correct_degree_for_cubic(self) -> None:
        """For x^3, selects level >= 3 and CV score at level 3 is near zero."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 3, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # levels 1-2 should have large error, levels >= 3 near zero
        cv_scores = result.cv_scores()
        bkd.assert_allclose(
            cv_scores[2:3], bkd.zeros(1), atol=1e-12
        )
        self.assertGreater(float(cv_scores[0]), 1e-2)
        self.assertGreaterEqual(result.best_label(), 3)

    def test_multivariate_degree_selection(self) -> None:
        """Selects correct level for 2D quadratic."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=4)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # f(x,y) = x^2 + y^2 -> degree 2 polynomial
        values = bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.best_label(), 2)

    def test_returns_cv_selection_result(self) -> None:
        """Fit returns CVSelectionResult."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars=1, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, CVSelectionResult)

    def test_cv_scores_shape(self) -> None:
        """CV scores array matches number of candidates."""
        bkd = self._bkd
        nvars = 1
        levels = [1, 2, 3, 4]
        expansion = self._create_expansion(nvars=1, max_level=4)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 30)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=levels, index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.cv_scores().shape[0], len(levels))

    def test_underdetermined_skipped(self) -> None:
        """Levels with too many terms for nsamples get inf score."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=10)
        ntrain = 8  # Very few samples
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(
            samples[0, :] + samples[1, :], (1, -1)
        )

        # Level 10 has (10+1)*(10+2)/2 = 66 terms >> 8 samples
        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 10], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # Level 10 should be inf, so level 1 is selected
        self.assertEqual(result.best_label(), 1)
        cv_scores = result.cv_scores()
        self.assertEqual(float(cv_scores[1]), float("inf"))

    def test_accessors(self) -> None:
        """Accessors return correct values."""
        bkd = self._bkd
        nvars = 1
        levels = [1, 2, 3]
        index_seq = HyperbolicIndexSequence(nvars, 0.5, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=levels, index_sequence=index_seq, alpha=1e-3
        )
        self.assertEqual(fitter.levels(), levels)
        self.assertIs(fitter.index_sequence(), index_seq)
        self.assertAlmostEqual(fitter.alpha(), 1e-3)

    def test_with_ridge_regularization(self) -> None:
        """Level selection works with ridge regularization."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5],
            index_sequence=index_seq, alpha=1e-6
        )
        result = fitter.fit(expansion, samples, values)

        # With small alpha, should still select level 2
        self.assertEqual(result.best_label(), 2)

    def test_with_lmo_nfolds(self) -> None:
        """Level selection works with LMO CV."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4, 5],
            index_sequence=index_seq, nfolds=5
        )
        result = fitter.fit(expansion, samples, values)

        # Level >= 2 should be selected (exact noiseless fit)
        self.assertGreaterEqual(result.best_label(), 2)
        # Level 1 should have significantly larger error
        self.assertGreater(
            float(result.cv_scores()[0]), float(result.cv_scores()[1]) + 1e-3
        )

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate from result evaluates correctly."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        # Evaluate on test samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))

        bkd.assert_allclose(predicted, expected, atol=1e-10)

    def test_empty_levels_raises(self) -> None:
        """Empty levels list raises ValueError."""
        bkd = self._bkd
        nvars = 1
        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        with self.assertRaises(ValueError):
            PCEDegreeSelectionFitter(
                bkd, levels=[], index_sequence=index_seq
            )

    def test_all_params_stored(self) -> None:
        """All candidate params are stored in result."""
        bkd = self._bkd
        nvars = 1
        levels = [1, 2, 3]
        expansion = self._create_expansion(nvars=1, max_level=3)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        index_seq = HyperbolicIndexSequence(nvars, 1.0, bkd)
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=levels, index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(len(result.all_params()), len(levels))

    def test_with_sparse_grid_index_sequence(self) -> None:
        """Level selection works with SparseGridIndexSequence."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=4)
        ntrain = 80
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # f(x,y) = x^2 + y^2 -> degree 2 polynomial
        values = bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )

        # Use LinearGrowthRule(1,1) so sparse grid level = total degree
        index_seq = SparseGridIndexSequence(
            nvars, bkd, growth_rules=LinearGrowthRule(scale=1, shift=1)
        )
        fitter = PCEDegreeSelectionFitter(
            bkd, levels=[1, 2, 3, 4], index_sequence=index_seq
        )
        result = fitter.fit(expansion, samples, values)

        self.assertEqual(result.best_label(), 2)


class TestPCEDegreeSelectionFitterNumpy(
    TestPCEDegreeSelectionFitter[NDArray[Any]]
):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestPCEDegreeSelectionFitterTorch(
    TestPCEDegreeSelectionFitter[torch.Tensor]
):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
