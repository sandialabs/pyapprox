"""Tests for operator-learning diagnostics."""

from typing import Any

import pytest
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis.orthonormal_poly import (
    OrthonormalPolynomialBasis,
)
from pyapprox.surrogates.affine.indices.utils import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate.factory import create_bases_1d
from pyapprox.surrogates.operatorlearning import (
    bochner_error,
    christoffel_integral,
    gram_condition_number,
    sample_complexity,
    weighted_gram,
)
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    m_orthonormality_drift,
)


def _legendre_basis(bkd: Backend, nvars: int, max_level: int) -> Any:
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    basis = OrthonormalPolynomialBasis(create_bases_1d(marginals, bkd), bkd)
    basis.set_indices(compute_hyperbolic_indices(nvars, max_level, 1.0, bkd))
    return basis


def _tensor_quadrature(bkd: Backend, basis: Any, npoints: int) -> Any:
    """Return (points, weights) for a tensor Gauss rule on the basis."""
    return basis.tensor_product_quadrature([npoints] * basis.nvars())


class TestWeightedGram:
    def test_unit_weights_match_unweighted(self, bkd: Backend) -> None:
        design = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        expected = weighted_gram(design, None, bkd)
        actual = weighted_gram(design, bkd.ones((3,)), bkd)
        bkd.assert_allclose(actual, expected)

    def test_shape(self, bkd: Backend) -> None:
        design = bkd.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert weighted_gram(design, None, bkd).shape == (2, 2)

    def test_matches_explicit_sum(self, bkd: Backend) -> None:
        design = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        weights = bkd.asarray([0.5, 1.5])
        expected = (
            0.5 * bkd.asarray([[1.0, 2.0], [2.0, 4.0]])
            + 1.5 * bkd.asarray([[9.0, 12.0], [12.0, 16.0]])
        ) / 2.0
        bkd.assert_allclose(weighted_gram(design, weights, bkd), expected)

    def test_orthonormal_basis_gives_identity_under_quadrature(
        self, bkd: Backend
    ) -> None:
        """T2: exact quadrature makes the weighted Gram exactly I.

        Uses the quadrature weights scaled by nsamples, since
        weighted_gram divides by the sample count.
        """
        basis = _legendre_basis(bkd, 2, 2)
        points, quad_weights = _tensor_quadrature(bkd, basis, 5)
        design = basis(points)
        nsamples = design.shape[0]
        gram = weighted_gram(design, quad_weights * nsamples, bkd)
        bkd.assert_allclose(gram, bkd.eye(basis.nterms()), atol=1e-12)

    def test_rejects_wrong_weight_shape(self, bkd: Backend) -> None:
        design = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="wrong shape"):
            weighted_gram(design, bkd.ones((3,)), bkd)


class TestGramConditionNumber:
    def test_identity_has_unit_condition(self, bkd: Backend) -> None:
        """An orthonormal design under exact quadrature is perfectly conditioned."""
        basis = _legendre_basis(bkd, 2, 2)
        points, quad_weights = _tensor_quadrature(bkd, basis, 5)
        design = basis(points)
        nsamples = design.shape[0]
        cond = gram_condition_number(
            design, quad_weights * nsamples, bkd
        )
        assert cond == pytest.approx(1.0, abs=1e-8)

    def test_rank_deficient_design_is_ill_conditioned(
        self, bkd: Backend
    ) -> None:
        """Duplicate columns leave the Gram singular."""
        design = bkd.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert gram_condition_number(design, None, bkd) > 1e12

    def test_at_least_one(self, bkd: Backend) -> None:
        design = bkd.asarray([[1.0, 0.5], [0.25, 2.0], [3.0, 1.0]])
        assert gram_condition_number(design, None, bkd) >= 1.0


class TestSampleComplexity:
    def test_grows_with_nterms(self, bkd: Backend) -> None:
        assert sample_complexity(10, 0.5, 0.5) < sample_complexity(
            100, 0.5, 0.5
        )

    def test_grows_as_delta_tightens(self, bkd: Backend) -> None:
        assert sample_complexity(10, 0.5, 0.5) < sample_complexity(
            10, 0.1, 0.5
        )

    def test_grows_as_epsilon_tightens(self, bkd: Backend) -> None:
        assert sample_complexity(10, 0.5, 0.5) < sample_complexity(
            10, 0.5, 0.01
        )

    def test_exceeds_nterms(self, bkd: Backend) -> None:
        """A stable fit always needs more samples than unknowns."""
        for nterms in (1, 10, 100):
            assert sample_complexity(nterms, 0.5, 0.5) > nterms

    def test_returns_int(self, bkd: Backend) -> None:
        assert isinstance(sample_complexity(10, 0.5, 0.5), int)

    @pytest.mark.parametrize("delta", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_delta(self, bkd: Backend, delta: float) -> None:
        with pytest.raises(ValueError, match="delta"):
            sample_complexity(10, delta, 0.5)

    @pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_epsilon(self, bkd: Backend, epsilon: float) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            sample_complexity(10, 0.5, epsilon)

    def test_rejects_nonpositive_nterms(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="nterms"):
            sample_complexity(0, 0.5, 0.5)


class TestOrthonormalityDrift:
    """T2, through the shared metric utility rather than a local copy."""

    def test_orthonormal_basis_under_exact_quadrature(
        self, bkd: Backend
    ) -> None:
        """T2: E_rho[p_i p_j] = delta_ij by tensor Gauss."""
        basis = _legendre_basis(bkd, 2, 3)
        points, weights = _tensor_quadrature(bkd, basis, 6)
        drift = m_orthonormality_drift(
            basis(points), DiagonalInnerProduct(weights, bkd), bkd
        )
        assert drift < 1e-12

    def test_detects_non_orthonormal_basis(self, bkd: Backend) -> None:
        """A monomial-like basis is not orthonormal under the measure."""
        basis_values = bkd.asarray(
            [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]
        )
        weights = bkd.full((4,), 0.25)
        drift = m_orthonormality_drift(
            basis_values, DiagonalInnerProduct(weights, bkd), bkd
        )
        assert drift > 0.1


class TestChristoffelIntegral:
    @pytest.mark.parametrize("max_level", [1, 2, 3])
    def test_integrates_to_one(self, bkd: Backend, max_level: int) -> None:
        """T2: int k_Lambda d_rho == 1 for any index set."""
        basis = _legendre_basis(bkd, 2, max_level)
        points, weights = _tensor_quadrature(bkd, basis, 6)
        integral = christoffel_integral(basis(points), weights, bkd)
        assert integral == pytest.approx(1.0, abs=1e-12)

    def test_detects_non_orthonormal_basis(self, bkd: Backend) -> None:
        """A basis of the wrong scale integrates away from one."""
        basis_values = bkd.asarray([[2.0], [2.0], [2.0], [2.0]])
        weights = bkd.full((4,), 0.25)
        assert christoffel_integral(basis_values, weights, bkd) == (
            pytest.approx(4.0)
        )


class TestBochnerError:
    def test_zero_for_exact_match(self, bkd: Backend) -> None:
        reference = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])
        assert bochner_error(reference, reference, bkd) == pytest.approx(0.0)

    def test_relative_scaling(self, bkd: Backend) -> None:
        """Doubling both prediction error and reference leaves it unchanged."""
        reference = bkd.asarray([[3.0, 4.0]])
        predicted = bkd.asarray([[3.0, 0.0]])
        single = bochner_error(predicted, reference, bkd)
        doubled = bochner_error(2 * predicted, 2 * reference, bkd)
        assert single == pytest.approx(doubled)

    def test_matches_hand_computation(self, bkd: Backend) -> None:
        reference = bkd.asarray([[3.0], [4.0]])
        predicted = bkd.asarray([[0.0], [0.0]])
        assert bochner_error(predicted, reference, bkd) == pytest.approx(1.0)

    def test_rejects_shape_mismatch(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="does not match"):
            bochner_error(
                bkd.asarray([[1.0]]), bkd.asarray([[1.0], [2.0]]), bkd
            )

    def test_rejects_zero_reference(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="all zero"):
            bochner_error(
                bkd.asarray([[1.0]]), bkd.zeros((1, 1)), bkd
            )
