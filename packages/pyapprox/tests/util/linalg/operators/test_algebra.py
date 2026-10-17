"""Conformance tests for the operator combinators on every stack."""

from typing import Any

import numpy as np
import pytest

from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, BackendVectorSpace
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
from tests._helpers.operator_stacks import StackCase, random_matrix, spd_matrix

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)

_NONE = OperatorProperties()
_SYM = OperatorProperties(symmetric=True)
_SPD = OperatorProperties(symmetric=True, spd=True)
_DIAG = OperatorProperties(symmetric=True, diagonal=True)
_DIAG_SPD = OperatorProperties(symmetric=True, spd=True, diagonal=True)
_IDENT = OperatorProperties(symmetric=True, spd=True, diagonal=True, identity=True)


def _dot_identity(case: StackCase, op: Any, seed: int) -> None:
    r"""<A x, y> = <x, A^T y>."""
    bkd = case.bkd
    x = case.vec(op.domain().size, 1, seed)
    y = case.vec(op.range().size, 1, seed + 1)
    lhs = bkd.reshape(bkd.sum(op.apply(x) * y), (1,))
    rhs = bkd.reshape(bkd.sum(x * op.apply_transpose(y)), (1,))
    bkd.assert_allclose(lhs, rhs, rtol=1e-12)


class TestScaledSum:
    def test_action_and_format_equal_the_matrix_sum(self, case: StackCase) -> None:
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
        x = case.vec(5, 2, 2)
        bkd.assert_allclose(op.apply(x), bkd.dot(expected, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(expected.T, x), rtol=1e-12)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), expected, rtol=1e-12)
        _dot_identity(case, op, 3)

    def test_format_is_computed_once(self, case: StackCase) -> None:
        op = ScaledSum([(1.0, case.identity(_U))], case.space(_U))
        assert op.format() is op.format()

    def test_apply_with_out(self, case: StackCase) -> None:
        bkd = case.bkd
        a = random_matrix(5, 5, 4)
        op = ScaledSum(
            [(1.0, case.build(a, _U)), (2.0, case.identity(_U))], case.space(_U)
        )
        x = case.vec(5, 1, 5)
        out = bkd.zeros((5, 1))
        result = op.apply(x, out=out)
        bkd.assert_allclose(result, bkd.dot(case.dense(a + 2.0 * np.eye(5)), x))
        assert (result is out) == case.in_place

    def test_rectangular_needs_both_spaces(self, case: StackCase) -> None:
        bkd = case.bkd
        a, b = random_matrix(3, 5, 6), random_matrix(3, 5, 7)
        terms = [(1.0, case.build(a, _U, _P)), (-1.0, case.build(b, _U, _P))]
        op = ScaledSum(terms, case.space(_P), case.space(_U))
        y = case.vec(3, 1, 8)
        bkd.assert_allclose(
            op.apply_transpose(y), bkd.dot(case.dense(a - b).T, y), rtol=1e-12
        )
        with pytest.raises(SpaceMismatchError, match="domain space"):
            ScaledSum(terms, case.space(_P))

    def test_rejects_mismatched_terms(self, case: StackCase) -> None:
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
    def test_action_format_and_layouts(self, case: StackCase) -> None:
        bkd = case.bkd
        a = random_matrix(3, 5, 9)
        op = Transposed(case.build(a, _U, _P))
        assert op.domain() == _P and op.range() == _U
        y = case.vec(3, 2, 10)
        bkd.assert_allclose(op.apply(y), bkd.dot(case.dense(a).T, y), rtol=1e-12)
        x = case.vec(5, 2, 11)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(case.dense(a), x))
        assert isinstance(op.format(), case.format_type)
        bkd.assert_allclose(case.format_dense(op.format()), case.dense(a).T)
        _dot_identity(case, op, 12)

    def test_keeps_properties(self, case: StackCase) -> None:
        op = case.build(spd_matrix(5, 13), _U, properties=_SPD)
        assert Transposed(op).properties() == _SPD


class TestProduct:
    def test_action_matches_the_matrix_product(self, case: StackCase) -> None:
        """(3x5)(5x5)(5x3): a 3x3 operator on p, applied right to left."""
        bkd = case.bkd
        a, b, c = random_matrix(3, 5, 14), spd_matrix(5, 15), random_matrix(5, 3, 16)
        op = Product([case.build(a, _U, _P), case.build(b, _U), case.build(c, _P, _U)])
        assert op.domain() == _P and op.range() == _P
        expected = case.dense(a @ b @ c)
        y = case.vec(3, 2, 17)
        bkd.assert_allclose(op.apply(y), bkd.dot(expected, y), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(y), bkd.dot(expected.T, y), rtol=1e-12)
        assert op.format() == MatrixFree()
        _dot_identity(case, op, 18)

    def test_apply_with_out(self, case: StackCase) -> None:
        bkd = case.bkd
        a = spd_matrix(5, 19)
        op = Product([case.build(a, _U), case.identity(_U)])
        x = case.vec(5, 1, 20)
        out = bkd.zeros((5, 1))
        result = op.apply(x, out=out)
        bkd.assert_allclose(result, bkd.dot(case.dense(a), x), rtol=1e-12)
        assert (result is out) == case.in_place

    def test_rejects_unchained_layouts(self, case: StackCase) -> None:
        with pytest.raises(SpaceMismatchError, match="product factor"):
            Product([case.identity(_U), case.identity(_P)])
        with pytest.raises(ValueError, match="at least one factor"):
            Product([])


class TestAsMatrixFree:
    def test_keeps_action_and_properties_erases_format(self, case: StackCase) -> None:
        bkd = case.bkd
        a = spd_matrix(5, 21)
        stored = case.build(a, _U, properties=_SPD)
        op = as_matrix_free(stored)
        assert isinstance(op, ActionOperatorProtocol)
        assert op.format() == MatrixFree()
        assert op.properties() == _SPD
        x = case.vec(5, 2, 22)
        bkd.assert_allclose(op.apply(x), stored.apply(x))
        bkd.assert_allclose(op.apply_transpose(x), stored.apply_transpose(x))

    def test_mixed_formats_sum_after_erasure(self, case: StackCase) -> None:
        """A stored operator plus a product (always matrix-free): the
        shape of a stepper block with an inverse inside."""
        bkd = case.bkd
        a, b = spd_matrix(5, 23), random_matrix(5, 5, 24)
        product = Product([case.build(b, _U), case.build(b, _U)])
        op = ScaledSum(
            [(1.0, as_matrix_free(case.build(a, _U))), (0.5, product)],
            case.space(_U),
        )
        assert op.format() == MatrixFree()
        x = case.vec(5, 1, 25)
        bkd.assert_allclose(
            op.apply(x), bkd.dot(case.dense(a + 0.5 * b @ b), x), rtol=1e-12
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


class TestBackendGraph:
    def test_scaled_sum_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        r"""d(y^T S x)/dx = S^T y through autograd, S = 2A - B."""
        import torch

        bkd = torch_bkd
        a, b = random_matrix(5, 5, 26), random_matrix(5, 5, 27)
        op = ScaledSum(
            [
                (2.0, BackendDenseOperator(bkd.asarray(a), _U, bkd)),
                (-1.0, BackendDenseOperator(bkd.asarray(b), _U, bkd)),
            ],
            BackendVectorSpace(_U, bkd),
        )
        rng = np.random.default_rng(28)
        x = torch.tensor(rng.normal(size=(5, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(5, 1)))
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)
