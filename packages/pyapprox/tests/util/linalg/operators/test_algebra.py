"""Conformance tests for the operator combinators on every stack."""

import numpy as np
import pytest

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    MatrixFree,
    OperatorProperties,
    Product,
    ScaledSum,
    SpaceDescriptor,
    SpaceMismatchError,
    Transposed,
    as_matrix_free,
    product_properties,
    scaled_sum_properties,
)
from tests._helpers.linalg.stacks import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)

_NONE = OperatorProperties()
_SYM = OperatorProperties(symmetric=True)
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG = OperatorProperties(symmetric=True, diagonal=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)
_IDENT = OperatorProperties(symmetric=True, spd=True, diagonal=True, identity=True)


class TestScaledSum:
    def test_action_and_format_equal_the_matrix_sum(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        bkd = case.bkd
        a, b = random_matrix(5, 5, 0), spd_matrix(5, 1)
        op = ScaledSum(
            [
                (2.0, case.build(a, _U)),
                (-0.5, case.build(b, _U)),
                (3.0, case.identity(_U)),
            ],
            case.space(_U),
        )
        expected = case.dense(2.0 * a - 0.5 * b + 3.0 * np.eye(5))
        x = case.vec(_U, 2, 2)
        case.assert_vector(op.apply(x), case.times(expected, x), rtol=1e-12)
        case.assert_vector(op.apply_transpose(x), case.times(expected.T, x), rtol=1e-12)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), expected, rtol=1e-12)
        case.assert_dot_identity(op, 3)

    def test_format_is_computed_once(self, case: TypedStackCase[V, F, A]) -> None:
        op = ScaledSum([(1.0, case.identity(_U))], case.space(_U))
        assert op.format() is op.format()

    def test_apply_with_out(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = random_matrix(5, 5, 4)
        op = ScaledSum(
            [(1.0, case.build(a, _U)), (2.0, case.identity(_U))], case.space(_U)
        )
        x = case.vec(_U, 1, 5)
        out = case.vector(bkd.zeros((5, 1)), _U)
        result = op.apply(x, out=out)
        case.assert_vector(result, case.times(case.dense(a + 2.0 * np.eye(5)), x))
        assert (result is out) == case.in_place

    def test_rectangular_needs_both_spaces(self, case: TypedStackCase[V, F, A]) -> None:
        a, b = random_matrix(3, 5, 6), random_matrix(3, 5, 7)
        terms = [(1.0, case.build(a, _U, _P)), (-1.0, case.build(b, _U, _P))]
        op = ScaledSum(terms, case.space(_P), case.space(_U))
        y = case.vec(_P, 1, 8)
        case.assert_vector(
            op.apply_transpose(y), case.times(case.dense(a - b).T, y), rtol=1e-12
        )
        with pytest.raises(SpaceMismatchError, match="domain space"):
            ScaledSum(terms, case.space(_P))

    def test_rejects_mismatched_terms(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(SpaceMismatchError, match="summed operator"):
            ScaledSum(
                [
                    (1.0, case.identity(_U)),
                    (1.0, case.identity(SpaceDescriptor("v", 5))),
                ],
                case.space(_U),
            )
        with pytest.raises(SpaceMismatchError, match="space of a scaled sum"):
            ScaledSum([(1.0, case.identity(_U))], case.space(SpaceDescriptor("v", 5)))
        with pytest.raises(ValueError, match="at least one term"):
            ScaledSum([], case.space(_U))


class TestTransposed:
    def test_action_format_and_layouts(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = random_matrix(3, 5, 9)
        op = Transposed(case.build(a, _U, _P))
        assert op.domain() == _P and op.range() == _U
        y = case.vec(_P, 2, 10)
        case.assert_vector(op.apply(y), case.times(case.dense(a).T, y), rtol=1e-12)
        x = case.vec(_U, 2, 11)
        case.assert_vector(op.apply_transpose(x), case.times(case.dense(a), x))
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), case.dense(a).T)
        case.assert_dot_identity(op, 12)

    def test_keeps_properties(self, case: TypedStackCase[V, F, A]) -> None:
        op = case.build(spd_matrix(5, 13), _U, properties=_SPD)
        assert Transposed(op).properties() == _SPD


class TestProduct:
    def test_action_matches_the_matrix_product(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """(3x5)(5x5)(5x3): a 3x3 operator on p, applied right to left."""
        a, b, c = random_matrix(3, 5, 14), spd_matrix(5, 15), random_matrix(5, 3, 16)
        op = Product([case.build(a, _U, _P), case.build(b, _U), case.build(c, _P, _U)])
        assert op.domain() == _P and op.range() == _P
        expected = case.dense(a @ b @ c)
        y = case.vec(_P, 2, 17)
        case.assert_vector(op.apply(y), case.times(expected, y), rtol=1e-12)
        case.assert_vector(op.apply_transpose(y), case.times(expected.T, y), rtol=1e-12)
        assert op.format() == MatrixFree()
        case.assert_dot_identity(op, 18)

    def test_apply_with_out(self, case: TypedStackCase[V, F, A]) -> None:
        bkd = case.bkd
        a = spd_matrix(5, 19)
        op = Product([case.build(a, _U), case.identity(_U)])
        x = case.vec(_U, 1, 20)
        out = case.vector(bkd.zeros((5, 1)), _U)
        result = op.apply(x, out=out)
        case.assert_vector(result, case.times(case.dense(a), x), rtol=1e-12)
        assert (result is out) == case.in_place

    def test_rejects_unchained_layouts(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(SpaceMismatchError, match="product factor"):
            Product([case.identity(_U), case.identity(_P)])
        with pytest.raises(ValueError, match="at least one factor"):
            Product([])


class TestAsMatrixFree:
    def test_keeps_action_and_properties_erases_format(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        a = spd_matrix(5, 21)
        stored = case.build(a, _U, properties=_SPD)
        op = as_matrix_free(stored)
        assert isinstance(op, ActionOperatorProtocol)
        assert op.format() == MatrixFree()
        assert op.properties() == _SPD
        x = case.vec(_U, 2, 22)
        case.assert_vector(op.apply(x), case.array(stored.apply(x)))
        case.assert_vector(op.apply_transpose(x), case.array(stored.apply_transpose(x)))

    def test_mixed_formats_sum_after_erasure(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """A stored operator plus a product (always matrix-free): the
        shape of a stepper block with an inverse inside."""
        a, b = spd_matrix(5, 23), random_matrix(5, 5, 24)
        product = Product([case.build(b, _U), case.build(b, _U)])
        op = ScaledSum(
            [(1.0, as_matrix_free(case.build(a, _U))), (0.5, product)],
            case.space(_U),
        )
        assert op.format() == MatrixFree()
        x = case.vec(_U, 1, 25)
        case.assert_vector(
            op.apply(x), case.times(case.dense(a + 0.5 * b @ b), x), rtol=1e-12
        )


class TestPropertyPropagation:
    @pytest.mark.parametrize(
        "terms, expected",
        [
            ([(1.0, _SYM), (-2.0, _SYM)], _SYM),
            ([(1.0, _SPD), (2.0, _SPD)], _SPD),
            ([(1.0, _SPD), (-2.0, _SPD)], _SYM),
            ([(1.0, _SPD), (1.0, _NONE)], _NONE),
            ([(1.0, _DIAG_SPD), (3.0, _IDENT)], _DIAG_SPD),
            ([(1.0, _IDENT)], _DIAG_SPD),
            ([(1.0, _DIAG), (1.0, _SPD)], _SYM),
        ],
    )
    def test_scaled_sum(
        self,
        terms: list[tuple[float, OperatorProperties]],
        expected: OperatorProperties,
    ) -> None:
        """A sum is never declared the identity, even of one identity."""
        assert scaled_sum_properties(terms) == expected

    @pytest.mark.parametrize(
        "factors, expected",
        [
            ([_SPD, _SPD], _NONE),
            ([_SYM, _DIAG], _NONE),
            ([_DIAG, _DIAG_SPD], _DIAG),
            ([_DIAG_SPD, _IDENT], _DIAG_SPD),
            ([_IDENT, _IDENT], _IDENT),
        ],
    )
    def test_product(
        self, factors: list[OperatorProperties], expected: OperatorProperties
    ) -> None:
        """A product of symmetric matrices is not symmetric in general."""
        assert product_properties(factors) == expected
