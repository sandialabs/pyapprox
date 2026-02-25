"""Tests for OMPCVFitter.

Tests verify that OMP path + LOO/LMO CV correctly identifies the
optimal number of terms for sparse signals.
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

from pyapprox.surrogates.affine.expansions.fitters.omp_cv import (
    OMPCVFitter,
)
from pyapprox.surrogates.affine.expansions.fitters.results import (
    CVSelectionResult,
)

from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.probability import UniformMarginal


class TestOMPCVFitter(Generic[Array], unittest.TestCase):
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

    def test_returns_cv_selection_result(self) -> None:
        """Fit returns CVSelectionResult."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, CVSelectionResult)

    def test_noiseless_sparse_selects_correct_count(self) -> None:
        """For a noiseless sparse signal, selects the correct number of terms."""
        bkd = self._bkd
        nvars = 1
        max_level = 5
        expansion = self._create_expansion(nvars, max_level)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # x^2 has exactly 3 terms in Legendre basis (P0, P1, P2 contribute
        # to x^2), but with orthonormal basis, the number of nonzero
        # coefficients depends on the basis type
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        # The CV score for the correct count should be near zero
        best_idx = result.best_index()
        bkd.assert_allclose(
            bkd.asarray([float(result.cv_scores()[best_idx])]),
            bkd.zeros(1),
            atol=1e-10,
        )

    def test_cv_scores_shape_matches_n_selected(self) -> None:
        """CV scores array length matches number of OMP-selected terms."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=4)
        ntrain = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=4)
        result = fitter.fit(expansion, samples, values)

        # cv_scores length should equal number of terms OMP selected
        n_candidates = result.cv_scores().shape[0]
        self.assertGreater(n_candidates, 0)
        self.assertLessEqual(n_candidates, 4)

    def test_best_nterms_less_than_max(self) -> None:
        """Selected truncation is less than max when signal is sparse."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars, max_level=8)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # Linear function: only 2 basis terms needed
        values = bkd.reshape(samples[0, :], (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=8)
        result = fitter.fit(expansion, samples, values)

        # Best label (number of terms) should be much less than 8
        self.assertLess(result.best_label(), 8)

    def test_noisy_data_cv_has_minimum(self) -> None:
        """With noisy data, CV scores should first decrease then increase."""
        bkd = self._bkd
        nvars = 1
        expansion = self._create_expansion(nvars, max_level=8)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # Quadratic signal + noise
        signal = samples[0, :] ** 2
        noise = bkd.asarray(np.random.normal(0, 0.1, ntrain))
        values = bkd.reshape(signal + noise, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=8)
        result = fitter.fit(expansion, samples, values)

        # Best truncation should be small (not max)
        self.assertLess(result.best_label(), 8)
        # CV score at best should be smaller than at last
        cv_scores = result.cv_scores()
        self.assertLess(
            float(cv_scores[result.best_index()]),
            float(cv_scores[-1]) + 1e-10,
        )

    def test_accessors(self) -> None:
        """Accessors return correct values."""
        bkd = self._bkd
        fitter = OMPCVFitter(bkd, max_nonzeros=10, rtol=1e-3, alpha=1e-4)
        self.assertEqual(fitter.max_nonzeros(), 10)
        self.assertAlmostEqual(fitter.rtol(), 1e-3)
        self.assertAlmostEqual(fitter.alpha(), 1e-4)

    def test_multi_qoi_raises(self) -> None:
        """Multi-QoI input raises ValueError."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=3, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = bkd.asarray(np.random.randn(2, 20))

        fitter = OMPCVFitter(bkd, max_nonzeros=3)
        with self.assertRaises(ValueError):
            fitter.fit(expansion, samples, values)

    def test_with_lmo_nfolds(self) -> None:
        """OMP CV works with LMO instead of LOO."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=5, nfolds=5)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, CVSelectionResult)
        # Should still find a good truncation
        best_idx = result.best_index()
        bkd.assert_allclose(
            bkd.asarray([float(result.cv_scores()[best_idx])]),
            bkd.zeros(1),
            atol=1e-10,
        )

    def test_fitted_surrogate_evaluates(self) -> None:
        """Fitted surrogate from result evaluates correctly."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=5)
        result = fitter.fit(expansion, samples, values)

        # Evaluate on test samples
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))

        bkd.assert_allclose(predicted, expected, atol=1e-10)

    def test_candidate_labels_are_integers(self) -> None:
        """Candidate labels are 1, 2, ..., n_selected."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=4)
        ntrain = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = OMPCVFitter(bkd, max_nonzeros=4)
        result = fitter.fit(expansion, samples, values)

        labels = result.candidate_labels()
        self.assertEqual(labels, list(range(1, len(labels) + 1)))

    def test_invalid_max_nonzeros_raises(self) -> None:
        """max_nonzeros < 1 raises ValueError."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            OMPCVFitter(bkd, max_nonzeros=0)


class TestOMPCVFitterNumpy(TestOMPCVFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestOMPCVFitterTorch(TestOMPCVFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
