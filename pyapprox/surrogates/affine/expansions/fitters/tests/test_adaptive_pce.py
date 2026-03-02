"""Tests for AdaptivePCEFitter.

Tests verify that the adaptive basis selection fitter (Jakeman et al. 2015)
correctly recovers sparse polynomial structures, respects constraints, and
maintains expected algorithm behavior.
"""

import numpy as np
import pytest

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


class TestAdaptivePCEFitter:
    """Base test class for AdaptivePCEFitter."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        """Create test expansion with orthonormal Legendre basis."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_fit_returns_adaptive_result(self, bkd) -> None:
        """Fit returns AdaptivePCEResult."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(expansion, samples, values)

        assert isinstance(result, AdaptivePCEResult)

    def test_recovers_known_pce(self, bkd) -> None:
        """Fitter recovers a target PCE with known coefficients."""
        nvars = 2

        # Create target PCE at degree 2 (total degree)
        target_expansion = self._create_expansion(bkd, nvars, max_level=2)
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
        fit_expansion = self._create_expansion(bkd, nvars, max_level=5)
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
        assert final_nterms <= true_nterms * 2

    def test_recovers_sparse_1d(self, bkd) -> None:
        """1D x^2 should converge with low CV."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=8)
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

    def test_recovers_sparse_2d(self, bkd) -> None:
        """2D x1^2 + x2^2 should find degree-2 terms."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=5)
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

    def test_stops_when_no_improvement(self, bkd) -> None:
        """Simple target should stop early, not run max_iterations."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=10)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        # Linear function: very simple
        values = bkd.reshape(samples[0, :], (1, -1))

        fitter = AdaptivePCEFitter(
            bkd, initial_level=2, max_level=10, max_iterations=50
        )
        result = fitter.fit(expansion, samples, values)

        # Should stop well before 50 iterations (init + at most a few outer)
        assert len(result.cv_scores_history()) < 50

    def test_num_expansions_controls_inner_loop(self, bkd) -> None:
        """num_expansions=1 vs 3 gives different inner loop behavior."""
        nvars = 2
        expansion1 = self._create_expansion(bkd, nvars, max_level=5)
        expansion3 = self._create_expansion(bkd, nvars, max_level=5)
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
        assert len(result1.cv_scores_history()) > 0
        assert len(result3.cv_scores_history()) > 0

    def test_downward_closure_maintained(self, bkd) -> None:
        """Final indices should be downward closed (since they are either
        the initial hyperbolic set or an expanded set, both of which are
        downward closed by construction)."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=5)
        ntrain = 150
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=5)
        result = fitter.fit(expansion, samples, values)

        final = result.final_indices()
        closure = compute_downward_closure(final, bkd)
        # Closure should have same number of indices as final
        assert final.shape[1] == closure.shape[1]

    def test_max_level_respected(self, bkd) -> None:
        """No index in the final set should exceed max_level."""
        nvars = 2
        max_level = 3
        expansion = self._create_expansion(bkd, nvars, max_level=5)
        ntrain = 150
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, ntrain)))
        values = bkd.reshape(samples[0, :] ** 3 + samples[1, :] ** 3, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=max_level)
        result = fitter.fit(expansion, samples, values)

        final = result.final_indices()
        # Check that total degree of each index <= max_level
        total_degrees = bkd.to_numpy(bkd.sum(final, axis=0))
        for td in total_degrees:
            assert int(td) <= max_level

    def test_max_iterations_respected(self, bkd) -> None:
        """Should not exceed max_iterations outer loops."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=10)
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
        assert len(result.cv_scores_history()) <= max_iter + 1

    def test_fitted_surrogate_evaluates(self, bkd) -> None:
        """Result evaluates correctly on test data."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=3, max_level=5)
        result = fitter.fit(expansion, samples, values)

        # Should be able to evaluate on new samples
        np.random.seed(99)
        test_samples = bkd.asarray(np.random.uniform(-1, 1, (1, 10)))
        predicted = result(test_samples)
        assert predicted.shape[0] == 1
        assert predicted.shape[1] == 10

        # Also verify surrogate accessor works
        predicted2 = result.surrogate()(test_samples)
        bkd.assert_allclose(predicted, predicted2, atol=1e-14)

    def test_nqoi_gt1_raises(self, bkd) -> None:
        """ValueError for nqoi > 1."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=3, nqoi=1)
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        values = bkd.asarray(np.random.randn(2, 20))

        fitter = AdaptivePCEFitter(bkd, initial_level=2, max_level=3)
        with pytest.raises(ValueError):
            fitter.fit(expansion, samples, values)

    def test_cv_history_length(self, bkd) -> None:
        """History has correct length matching iterations run."""
        expansion = self._create_expansion(bkd, nvars=1, max_level=5)
        ntrain = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (1, ntrain)))
        values = bkd.reshape(samples[0, :] ** 2, (1, -1))

        fitter = AdaptivePCEFitter(bkd, initial_level=3, max_level=5)
        result = fitter.fit(expansion, samples, values)

        # All histories should have the same length
        n_iters = len(result.cv_scores_history())
        assert len(result.nterms_history()) == n_iters
        assert len(result.indices_history()) == n_iters
        assert n_iters > 0

    def test_accessors(self, bkd) -> None:
        """Constructor params accessible via accessors."""
        fitter = AdaptivePCEFitter(
            bkd,
            initial_level=2,
            pnorm=0.5,
            num_expansions=2,
            max_level=15,
            max_iterations=25,
            restrict_tol=1e-8,
        )
        assert fitter.initial_level() == 2
        assert fitter.pnorm() == pytest.approx(0.5)
        assert fitter.num_expansions() == 2
        assert fitter.max_level() == 15
        assert fitter.max_iterations() == 25
        assert fitter.restrict_tol() == pytest.approx(1e-8)

    def test_expand_indices_adds_admissible(self, bkd) -> None:
        """_expand_indices adds all admissible forward neighbors."""
        # Start with {(0,0), (1,0), (0,1)} in 2D
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]], dtype=bkd.int64_dtype())
        expanded = _expand_indices(indices, max_level=5, pnorm=1.0, bkd=bkd)

        # (2,0) is admissible: backward (1,0) exists
        # (0,2) is admissible: backward (0,1) exists
        # (1,1) is admissible: backward (0,1) and (1,0) both exist
        assert expanded.shape[1] == 6  # original 3 + 3 new

    def test_expand_indices_respects_max_level(self, bkd) -> None:
        """_expand_indices respects max_level constraint."""
        indices = bkd.asarray([[0, 1, 0], [0, 0, 1]], dtype=bkd.int64_dtype())
        expanded = _expand_indices(indices, max_level=1, pnorm=1.0, bkd=bkd)

        # No expansion possible: all forward neighbors exceed level 1
        assert expanded.shape[1] == 3

    def test_restrict_shrinks_basis(self, bkd) -> None:
        """Algorithm restricts to nonzero terms between iterations."""
        nvars = 2
        expansion = self._create_expansion(bkd, nvars, max_level=5)
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
