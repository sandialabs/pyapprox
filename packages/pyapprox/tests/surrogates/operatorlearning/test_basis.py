"""Tests for the separable operator basis."""

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
from pyapprox.surrogates.operatorlearning import SeparableOperatorBasis
from pyapprox.util.backends.protocols import Backend


def _scalar_basis(bkd: Backend, nvars: int, max_level: int) -> Any:
    """Legendre basis over [-1, 1]^nvars with a total-degree index set."""
    marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
    basis = OrthonormalPolynomialBasis(create_bases_1d(marginals, bkd), bkd)
    basis.set_indices(compute_hyperbolic_indices(nvars, max_level, 1.0, bkd))
    return basis


class TestSeparableOperatorBasis:
    def test_neffective_is_scalar_basis_size(self, bkd: Backend) -> None:
        scalar = _scalar_basis(bkd, 2, 2)
        operator = SeparableOperatorBasis(scalar, 7, bkd)
        assert operator.neffective() == scalar.nterms() == 6

    def test_nterms_multiplies_by_noutputs(self, bkd: Backend) -> None:
        """The full basis has one element per (index, output) pair."""
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 7, bkd)
        assert operator.nterms() == 6 * 7

    def test_design_matrix_shape(self, bkd: Backend) -> None:
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.0]])
        assert operator.design_matrix(coefs).shape == (3, 6)

    def test_christoffel_matches_definition(self, bkd: Backend) -> None:
        """T5: k_Lambda == (1 / N_eff) sum_lambda p_lambda^2."""
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 3), 4, bkd)
        coefs = bkd.asarray([[0.1, -0.7, 0.3], [-0.4, 0.5, 0.9]])
        design = operator.design_matrix(coefs)
        expected = bkd.sum(design**2, axis=1) / operator.neffective()
        bkd.assert_allclose(operator.christoffel(coefs), expected, rtol=1e-14)

    @pytest.mark.parametrize("noutputs", [1, 3, 17])
    def test_christoffel_independent_of_noutputs(
        self, bkd: Backend, noutputs: int
    ) -> None:
        """T5: d_out cancels exactly from the sampling weight.

        This is the identity that makes the sample count depend only on
        N_eff, so it is asserted across a wide range of d_out.
        """
        scalar = _scalar_basis(bkd, 2, 3)
        coefs = bkd.asarray([[0.1, -0.7, 0.3], [-0.4, 0.5, 0.9]])
        reference = SeparableOperatorBasis(scalar, 1, bkd).christoffel(coefs)
        actual = SeparableOperatorBasis(scalar, noutputs, bkd).christoffel(
            coefs
        )
        bkd.assert_allclose(actual, reference, rtol=1e-14)

    def test_neffective_independent_of_noutputs(self, bkd: Backend) -> None:
        scalar = _scalar_basis(bkd, 2, 3)
        assert (
            SeparableOperatorBasis(scalar, 1, bkd).neffective()
            == SeparableOperatorBasis(scalar, 50, bkd).neffective()
        )

    def test_christoffel_is_positive(self, bkd: Backend) -> None:
        """The constant term alone makes the sum strictly positive."""
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.0, 0.5], [0.0, -0.5]])
        assert bool(bkd.min(operator.christoffel(coefs)) > 0.0)

    def test_apply_matches_manual_product(self, bkd: Backend) -> None:
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.1, 0.2], [-0.4, 0.5]])
        params = bkd.asarray(
            [
                [1.0, 0.0, -1.0],
                [0.5, 2.0, 0.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.5],
                [0.25, -0.5, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        expected = bkd.dot(operator.design_matrix(coefs), params).T
        bkd.assert_allclose(operator.apply(params, coefs), expected)

    def test_apply_output_shape(self, bkd: Backend) -> None:
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.1, 0.2, 0.3, 0.4], [-0.4, 0.5, 0.0, 0.1]])
        params = bkd.zeros((6, 3))
        assert operator.apply(params, coefs).shape == (3, 4)

    def test_apply_rejects_wrong_param_shape(self, bkd: Backend) -> None:
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.1], [0.2]])
        with pytest.raises(ValueError, match="wrong shape"):
            operator.apply(bkd.zeros((6, 2)), coefs)

    def test_set_indices_changes_neffective(self, bkd: Backend) -> None:
        """Lambda is replaceable without rebuilding anything else."""
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        assert operator.neffective() == 6
        operator.set_indices(compute_hyperbolic_indices(2, 3, 1.0, bkd))
        assert operator.neffective() == 10

    def test_design_matrix_follows_new_indices(self, bkd: Backend) -> None:
        """The design matrix is recomputed, never cached at fit time."""
        operator = SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 3, bkd)
        coefs = bkd.asarray([[0.1, 0.2], [-0.4, 0.5]])
        assert operator.design_matrix(coefs).shape == (2, 6)
        operator.set_indices(compute_hyperbolic_indices(2, 3, 1.0, bkd))
        assert operator.design_matrix(coefs).shape == (2, 10)

    def test_get_indices_delegates(self, bkd: Backend) -> None:
        scalar = _scalar_basis(bkd, 2, 2)
        operator = SeparableOperatorBasis(scalar, 3, bkd)
        bkd.assert_allclose(operator.get_indices(), scalar.get_indices())

    def test_rejects_non_basis(self, bkd: Backend) -> None:
        with pytest.raises(TypeError, match="EvaluableMultiIndexBasisProtocol"):
            SeparableOperatorBasis("not_a_basis", 3, bkd)

    def test_rejects_nonpositive_noutputs(self, bkd: Backend) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            SeparableOperatorBasis(_scalar_basis(bkd, 2, 2), 0, bkd)
