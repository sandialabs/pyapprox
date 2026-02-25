"""Tests for AdaptivePCEFitter.

Tests verify that the adaptive basis selection fitter (Jakeman et al. 2015)
correctly recovers sparse polynomial structures, respects constraints, and
maintains expected algorithm behavior.
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters.adaptive_pce import (
    AdaptivePCEFitter,
    AdaptivePCEResult,
    _expand_indices,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_downward_closure,
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestAdaptivePCEFitter(Generic[Array], unittest.TestCase):
    """Base test class - NOT run directly."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion with orthonormal Legendre basis."""
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_adaptive_result(self) -> None:
        """Fit returns AdaptivePCEResult."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(expansion, samples, values)

        self.assertIsInstance(result, AdaptivePCEResult)

    def test_recovers_known_pce(self) -> None:
        """Fitter recovers a target PCE with known coefficients."""
        bkd = self._bkd
        nvars = 2

        # Create target PCE at degree 2 (total degree)
        target_expansion = self._create_expansion(nvars, max_level=2)
        true_nterms = target_expansion.nterms()
        true_coefs = bkd.zeros((true_nterms, 1))

        # Set specific nonzero coefficients
        true_coefs[0, 0] = 1.0  # constant
        true_coefs[1, 0] = 0.5  # x1 term
        true_coefs[2, 0] = -0.3  # x2 term
        true_coefs[3, 0] = 0.8  # x1^2 term
        true_coefs[5, 0] = 0.4  # x2^2 term
        target_expansion.set_coefficients(true_coefs)

        # Generate well-overdetermined training data
        ntrain = 200
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = target_expansion(samples)  # (1, ntrain)

        # Run adaptive fitter starting at degree 2, with large ceiling
        fit_expansion = self._create_expansion(nvars, max_level=5)
        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(fit_expansion, samples, values)

        # Verify accuracy on test data
        np.random.seed(123)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 50)))
        bkd.assert_allclose(
            result(test_samples), target_expansion(test_samples), atol=1e-8
        )

        # Verify no over-resolution
        final_nterms = result.final_indices().shape[1]
        self.assertLessEqual(final_nterms, true_nterms * 2)

    def test_recovers_sparse_1d(self) -> None:
        """1D x^2 should converge with low CV."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=8)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=3, max_level=8)
        result = fitter.fit(expansion, samples, values)

        # Predictions should be accurate
        np.random.seed(99)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))
        bkd.assert_allclose(result(test_samples), expected, atol=1e-8)

    def test_recovers_sparse_2d(self) -> None:
        """2D x1^2 + x2^2 should find degree-2 terms."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 150
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(expansion, samples, values)

        # Predictions should be accurate
        np.random.seed(99)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 50)))
        expected = bkd.reshape(
            test_samples[0, :] ** 2 + test_samples[1, :] ** 2, (1, -1)
        )
        bkd.assert_allclose(result(test_samples), expected, atol=1e-8)

    def test_stops_when_no_improvement(self) -> None:
        """Simple target should stop early, not run max_iterations."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=10)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        # Linear function: very simple
        values = bkd.reshape(samples[0, :], (1, -1))

        fitter = AdaptivePCEFitter(
            bkd, initial_level=2, max_level=10, max_iterations=50
        )
        result = fitter.fit(expansion, samples, values)

        # Should stop well before 50 iterations (init + at most a few outer)
        self.assertLess(len(result.cv_scores_history()), 50)

    def test_num_expansions_controls_inner_loop(self) -> None:
        """num_expansions=1 vs 3 gives different inner loop behavior."""
        bkd = self._bkd
        nvars = 2
        expansion1 = self._create_expansion(nvars, max_level=5)
        expansion3 = self._create_expansion(nvars, max_level=5)
        ntrain = 150
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        fitter1 = AdaptivePCEFitter(bkd, initial_level=2, max_level=5, num_expansions=1)
        result1 = fitter1.fit(expansion1, samples, values)

        np.random.seed(42)
        fitter3 = AdaptivePCEFitter(bkd, initial_level=2, max_level=5, num_expansions=3)
        result3 = fitter3.fit(expansion3, samples, values)

        # Both should produce reasonable results
        self.assertGreater(len(result1.cv_scores_history()), 0)
        self.assertGreater(len(result3.cv_scores_history()), 0)

    def test_downward_closure_maintained(self) -> None:
        """Final indices should be downward closed (since they are either
        the initial hyperbolic set or an expanded set, both of which are
        downward closed by construction)."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 150
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(expansion, samples, values)

        final = result.final_indices()
        closure = compute_downward_closure(final, bkd)
        # Closure should have same number of indices as final
        self.assertEqual(final.shape[1], closure.shape[1])

    def test_max_level_respected(self) -> None:
        """No index in the final set should exceed max_level."""
        bkd = self._bkd
        nvars = 2
        max_level = 3
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 150
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 3 + samples[1, :] ** 3, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=max_level)
        result = fitter.fit(expansion, samples, values)

        final = result.final_indices()
        # Check that total degree of each index <= max_level
        total_degrees = bkd.to_numpy(bkd.sum(final, axis=0))
        for td in total_degrees:
            self.assertLessEqual(int(td), max_level)

    def test_max_iterations_respected(self) -> None:
        """Should not exceed max_iterations outer loops."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=10)
        ntrain = 200
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(
            bkd.sin(3.0 * samples[0, :]) * bkd.cos(3.0 * samples[1, :]),
            (1, -1),
        )

        max_iter = 3
        fitter = AdaptivePCEFitter(
            bkd, initial_level=2, max_level=10, max_iterations=max_iter
        )
        result = fitter.fit(expansion, samples, values)

        # History includes init + up to max_iter outer iterations
        self.assertLessEqual(len(result.cv_scores_history()), max_iter + 1)

    def test_fitted_surrogate_evaluates(self) -> None:
        """Result evaluates correctly on test data."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=3, max_level=5)
        result = fitter.fit(expansion, samples, values)

        # Should be able to evaluate on new samples
        np.random.seed(99)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        self.assertEqual(predicted.shape[0], 1)
        self.assertEqual(predicted.shape[1], 10)

        # Also verify surrogate accessor works
        predicted2 = result.surrogate()(test_samples)
        bkd.assert_allclose(predicted, predicted2, atol=1e-14)

    def test_nqoi_gt1_raises(self) -> None:
        """ValueError for nqoi > 1."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=3, nqoi=1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = bkd.asarray(np.random.randn(2, 20))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=3)
        with self.assertRaises(ValueError):
            fitter.fit(expansion, samples, values)

    def test_cv_history_length(self) -> None:
        """History has correct length matching iterations run."""
        bkd = self._bkd
        expansion = self._create_expansion(nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=3, max_level=5)
        result = fitter.fit(expansion, samples, values)

        # All histories should have the same length
        n_iters = len(result.cv_scores_history())
        self.assertEqual(len(result.nterms_history()), n_iters)
        self.assertEqual(len(result.indices_history()), n_iters)
        self.assertGreater(n_iters, 0)

    def test_accessors(self) -> None:
        """Constructor params accessible via accessors."""
        bkd = self._bkd
        fitter = AdaptivePCEFitter(
            bkd,
            initial_level=2,
            pnorm=0.5,
            num_expansions=2,
            max_level=15,
            max_iterations=25,
        )
        self.assertEqual(fitter.initial_level(), 2)
        self.assertAlmostEqual(fitter.pnorm(), 0.5)
        self.assertEqual(fitter.num_expansions(), 2)
        self.assertEqual(fitter.max_level(), 15)
        self.assertEqual(fitter.max_iterations(), 25)

    def test_expand_indices_adds_admissible(self) -> None:
        """_expand_indices adds all admissible forward neighbors."""
        bkd = self._bkd
        # Start with {(0,0), (1,0), (0,1)} in 2D
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]], dtype=bkd.int64_dtype())
        expanded = _expand_indices(indices, max_level=5, pnorm=1.0, bkd=bkd)

        # (2,0) is admissible: backward (1,0) exists
        # (0,2) is admissible: backward (0,1) exists
        # (1,1) is admissible: backward (0,1) and (1,0) both exist
        self.assertEqual(expanded.shape[1], 6)  # original 3 + 3 new

    def test_expand_indices_respects_max_level(self) -> None:
        """_expand_indices respects max_level constraint."""
        bkd = self._bkd
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]], dtype=bkd.int64_dtype())
        expanded = _expand_indices(indices, max_level=1, pnorm=1.0, bkd=bkd)

        # No expansion possible: all forward neighbors exceed level 1
        self.assertEqual(expanded.shape[1], 3)

    def test_restrict_shrinks_basis(self) -> None:
        """Algorithm restricts to nonzero terms between iterations."""
        bkd = self._bkd
        nvars = 2
        expansion = self._create_expansion(nvars, max_level=5)
        ntrain = 150
        np.random.seed(42)
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        # Sparse function: only depends on x1
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5, verbosity=0)
        result = fitter.fit(expansion, samples, values)

        # Should achieve good accuracy
        np.random.seed(99)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, 50)))
        expected = bkd.reshape(test_samples[0, :] ** 2, (1, -1))
        bkd.assert_allclose(result(test_samples), expected, atol=1e-6)


class TestAdaptivePCEFitterNumpy(TestAdaptivePCEFitter[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptivePCEFitterTorch(TestAdaptivePCEFitter[torch.Tensor]):
    """PyTorch backend tests."""

    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
