"""Tests for the basis-as-operations abstraction.

Two things are worth testing and one is not. Not worth testing: that
:class:`ArrayBasis` computes ``V c`` correctly, which is one ``dot`` and
would be testing the backend. Worth testing: that the operations are the
right *set* -- every use of a basis in this package goes through them --
and that they **compose**, since a consumer that selects columns and then
scales them is the case a streaming implementation must defer to block
time and a resident one gets right by accident.
"""

from typing import Any

import numpy as np
import pytest
from pyapprox.surrogates.kle.basis_operator import (
    ArrayBasis,
    BasisOperatorProtocol,
    as_basis_operator,
)
from pyapprox.util.backends.protocols import Backend


def _basis(bkd: Backend, nstates: int = 40, nterms: int = 6) -> Any:
    """Orthonormal, so a scaling is visible against a known norm."""
    raw = np.random.RandomState(0).standard_normal((nstates, nterms))
    return bkd.array(np.linalg.qr(raw)[0])


class TestArrayBasisSatisfiesTheProtocol:
    def test_is_runtime_checkable(self, bkd: Backend) -> None:
        """So a public boundary can reject a mis-typed argument."""
        assert isinstance(
            ArrayBasis(_basis(bkd), bkd), BasisOperatorProtocol
        )

    def test_reports_its_shape_under_both_namings(
        self, bkd: Backend
    ) -> None:
        """``nstates``/``nterms`` here, ``nrows``/``ncols`` for operators.

        The same two integers. Both names exist so a basis can feed the
        matrix-free machinery in ``util.linalg.randomized`` without an
        adapter, while reading as a basis everywhere else.
        """
        basis = ArrayBasis(_basis(bkd, 40, 6), bkd)
        assert (basis.nstates(), basis.nterms()) == (40, 6)
        assert (basis.nrows(), basis.ncols()) == (40, 6)

    def test_rejects_a_1d_basis(self, bkd: Backend) -> None:
        """A single vector someone forgot to give a second axis."""
        with pytest.raises(ValueError, match="2D"):
            ArrayBasis(bkd.array([1.0, 2.0]), bkd)

    def test_rejects_a_scaling_of_the_wrong_shape(
        self, bkd: Backend
    ) -> None:
        """One factor per term, and a length-one vector is not that.

        A ``(nterms,)`` vector and a length-one one both broadcast
        against a ``(nstates, nterms)`` basis, the second applying a
        single factor to every column. That produces an array of the
        right shape and the wrong content, so the length is checked
        rather than left to broadcasting.
        """
        basis = ArrayBasis(_basis(bkd, 40, 6), bkd)
        with pytest.raises(ValueError, match="one entry per term"):
            basis.scale(bkd.array([2.0]))


class TestTheOperationsCoverEveryUse:
    """The four things this package does with a basis, and no more.

    Catalogued from the call sites: column selection
    (``eig_vecs[:, :nterms]``, ``eigenvectors[:, selected]``), column
    scaling (``eig_vecs * sqrt_eig_vals``), contraction over rows
    (``dot(basis().T, f)``, ``dot(basis(), c)``, ``dot(basis()**2, v)``)
    and the dimension query. Each test below is one of those, checked
    against the array expression it replaces.
    """

    def test_select_matches_column_indexing(self, bkd: Backend) -> None:
        array = _basis(bkd, 40, 6)
        chosen = [4, 0, 3]
        bkd.assert_allclose(
            ArrayBasis(array, bkd).select(chosen).to_array(),
            array[:, bkd.asarray(chosen, dtype=int)],
            atol=0.0,
        )

    def test_select_preserves_the_order_it_is_given(
        self, bkd: Backend
    ) -> None:
        """Greedy selection yields indices in selection order, not sorted.

        A ``select`` that sorted would permute the latent coordinates
        against the coefficients fitted for them -- silently, since the
        result still has the right shape.
        """
        array = _basis(bkd, 40, 6)
        descending = ArrayBasis(array, bkd).select([5, 2, 0]).to_array()
        bkd.assert_allclose(descending[:, 0], array[:, 5], atol=0.0)
        bkd.assert_allclose(descending[:, 2], array[:, 0], atol=0.0)

    def test_scale_matches_column_multiplication(
        self, bkd: Backend
    ) -> None:
        array = _basis(bkd, 40, 6)
        factors = bkd.array(np.arange(1.0, 7.0))
        bkd.assert_allclose(
            ArrayBasis(array, bkd).scale(factors).to_array(),
            array * factors,
            atol=0.0,
        )

    def test_scale_is_per_term_not_per_state(self, bkd: Backend) -> None:
        """The orientation broadcasting would get wrong without erroring.

        Checked against an explicit per-column construction rather than
        another vectorized expression, which could share the mistake.
        """
        array = _basis(bkd, 40, 6)
        factors = bkd.array(np.arange(1.0, 7.0))
        raw = bkd.to_numpy(array)
        expected = np.stack(
            [raw[:, j] * (j + 1.0) for j in range(6)], axis=1
        )
        bkd.assert_allclose(
            ArrayBasis(array, bkd).scale(factors).to_array(),
            bkd.array(expected),
            atol=1e-14,
        )

    def test_square_matches_elementwise_squaring(
        self, bkd: Backend
    ) -> None:
        """What variance propagation contracts against."""
        array = _basis(bkd, 40, 6)
        bkd.assert_allclose(
            ArrayBasis(array, bkd).square().to_array(),
            array**2,
            atol=0.0,
        )

    def test_apply_matches_the_decode_product(self, bkd: Backend) -> None:
        array = _basis(bkd, 40, 6)
        coefs = bkd.array(
            np.random.RandomState(1).standard_normal((6, 5))
        )
        bkd.assert_allclose(
            ArrayBasis(array, bkd).apply(coefs),
            bkd.dot(array, coefs),
            atol=0.0,
        )

    def test_apply_transpose_matches_the_encode_product(
        self, bkd: Backend
    ) -> None:
        array = _basis(bkd, 40, 6)
        fields = bkd.array(
            np.random.RandomState(2).standard_normal((40, 5))
        )
        bkd.assert_allclose(
            ArrayBasis(array, bkd).apply_transpose(fields),
            bkd.dot(array.T, fields),
            atol=0.0,
        )


class TestOperationsCompose:
    """The property a streaming implementation has to get right.

    Selecting then scaling before contracting is what ``DataDrivenKLE``
    does: truncate to ``nterms``, multiply by ``sqrt(eigenvalues)``, then
    encode or decode. A resident implementation composes correctly
    whatever the order, so these are cheap here -- and they are the tests
    that will fail first against a streaming implementation that
    materializes at the wrong moment.
    """

    def test_select_then_scale_equals_the_array_expression(
        self, bkd: Backend
    ) -> None:
        array = _basis(bkd, 40, 6)
        chosen = [3, 1, 5]
        factors = bkd.array([2.0, 0.5, 4.0])
        composed = ArrayBasis(array, bkd).select(chosen).scale(factors)
        expected = array[:, bkd.asarray(chosen, dtype=int)] * factors
        bkd.assert_allclose(composed.to_array(), expected, atol=0.0)

    def test_a_composed_basis_contracts_correctly(
        self, bkd: Backend
    ) -> None:
        """The composition is only useful if it survives to the contraction."""
        array = _basis(bkd, 40, 6)
        chosen = [3, 1, 5]
        factors = bkd.array([2.0, 0.5, 4.0])
        fields = bkd.array(
            np.random.RandomState(3).standard_normal((40, 4))
        )
        composed = ArrayBasis(array, bkd).select(chosen).scale(factors)
        expected = array[:, bkd.asarray(chosen, dtype=int)] * factors
        bkd.assert_allclose(
            composed.apply_transpose(fields),
            bkd.dot(expected.T, fields),
            atol=1e-14,
        )

    def test_selecting_twice_narrows_against_the_current_columns(
        self, bkd: Backend
    ) -> None:
        """Indices address the basis as it stands, not the original.

        The alternative would make a restricted basis leaky: a consumer
        would need to know what it had been restricted from before it
        could index it.
        """
        array = _basis(bkd, 40, 6)
        twice = ArrayBasis(array, bkd).select([4, 2, 0]).select([1])
        assert twice.nterms() == 1
        bkd.assert_allclose(
            twice.to_array()[:, 0], array[:, 2], atol=0.0
        )

    def test_the_source_basis_is_not_mutated(self, bkd: Backend) -> None:
        """So a basis can be handed on without a defensive copy.

        At the sizes this exists for a defensive copy is gigabytes, so
        immutability is a memory property as much as a correctness one.
        """
        array = _basis(bkd, 40, 6)
        basis = ArrayBasis(array, bkd)
        before = bkd.array(np.array(bkd.to_numpy(basis.to_array())))
        basis.select([0, 1]).scale(bkd.array([3.0, 3.0])).square()
        bkd.assert_allclose(basis.to_array(), before, atol=0.0)


class TestTheAdapter:
    def test_wraps_a_bare_array(self, bkd: Backend) -> None:
        """So widening a signature costs its callers nothing."""
        assert isinstance(
            as_basis_operator(_basis(bkd), bkd), BasisOperatorProtocol
        )

    def test_passes_an_operator_through_unchanged(
        self, bkd: Backend
    ) -> None:
        """Wrapping one twice would bury a streaming basis in an array."""
        basis = ArrayBasis(_basis(bkd), bkd)
        assert as_basis_operator(basis, bkd) is basis
