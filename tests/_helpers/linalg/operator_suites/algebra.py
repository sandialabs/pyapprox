"""Stack-generic operator suites: ``ScaledSumSuite``,
``BlockAssembledSuite``, ``TransposedSuite``,
``TransposeCapabilitySuite``, ``ProductSuite``, ``AsMatrixFreeSuite``.

Each runs on the stack its ``case`` fixture supplies
(``TypedStackCase``); a stack's test directory subclasses it and
defines the fixture.
"""

from typing import Generic, Optional, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    BlockAssembled,
    FormatProtocol,
    InnerProductSpaceProtocol,
    LinearMapProtocol,
    MatrixFree,
    OperatorProperties,
    Product,
    ScaledSum,
    SpaceDescriptor,
    SpaceMismatchError,
    Transposed,
    as_matrix_free,
)
from tests._helpers.linalg.forward_only import ForwardOnly
from tests._helpers.linalg.stack_case import (
    A,
    F,
    TypedStackCase,
    V,
    random_matrix,
    spd_matrix,
)

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)
_SPD = OperatorProperties(symmetric=True, spd=True)


class ScaledSumSuite:
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

    @pytest.mark.parametrize("stored_first", [True, False], ids=["stored", "free"])
    def test_mixed_formats_apply_but_have_no_format(
        self, case: TypedStackCase[V, F, A], stored_first: bool
    ) -> None:
        """Either order: the action works, asking for storage is refused
        clearly (also by a direct solver), and as_matrix_free uses it."""
        a, b = random_matrix(5, 5, 9), random_matrix(5, 5, 10)
        stored = case.build(a, _U)
        free = as_matrix_free(case.build(b, _U))
        pair: list[tuple[float, ActionOperatorProtocol[V, FormatProtocol]]] = [
            (1.0, stored),
            (2.0, free),
        ]
        terms = pair if stored_first else pair[::-1]
        op = ScaledSum(terms, case.space(_U))
        x = case.vec(_U, 1, 11)
        case.assert_vector(
            op.apply(x), case.times(case.dense(a + 2.0 * b), x), rtol=1e-12
        )
        with pytest.raises(TypeError, match="different formats"):
            op.format()
        with pytest.raises(TypeError, match="different formats"):
            case.direct_solver().setup(op)  # type: ignore[arg-type]
        assert as_matrix_free(op).format() == MatrixFree()
        case.assert_vector(
            as_matrix_free(op).apply(x),
            case.times(case.dense(a + 2.0 * b), x),
            rtol=1e-12,
        )


class _ArrayPartition(Generic[V, F, A]):
    """Rows split and joined through a case's array boundary: a row
    partition for any stack the case covers."""

    def __init__(
        self,
        case: TypedStackCase[V, F, A],
        whole: SpaceDescriptor,
        parts: Sequence[SpaceDescriptor],
    ) -> None:
        self._case, self._whole, self._descriptors = case, whole, tuple(parts)
        self._offsets = np.cumsum([0] + [d.size for d in parts])

    def space(self) -> InnerProductSpaceProtocol[V, A]:
        return self._case.space(self._whole)

    def parts(self) -> list[InnerProductSpaceProtocol[V, A]]:
        return [self._case.space(d) for d in self._descriptors]

    def split(self, x: V) -> list[V]:
        array = self._case.array(x)
        return [
            self._case.vector(array[int(lo) : int(hi)], d)
            for lo, hi, d in zip(
                self._offsets[:-1], self._offsets[1:], self._descriptors
            )
        ]

    def join(self, parts: Sequence[V]) -> V:
        arrays = [self._case.array(part) for part in parts]
        return self._case.vector(self._case.bkd.vstack(arrays), self._whole)


_W = SpaceDescriptor("w", 5)
_W1, _W2 = SpaceDescriptor("w1", 3), SpaceDescriptor("w2", 2)


class BlockAssembledSuite:
    def _blocks(
        self, case: TypedStackCase[V, F, A]
    ) -> tuple[BlockAssembled[V, F], NDArray[np.float64]]:
        upper_left, lower_left = spd_matrix(3, 20), random_matrix(2, 3, 21)
        lower_right = random_matrix(2, 2, 22)
        grid: list[list[Optional[ActionOperatorProtocol[V, F]]]] = [
            [case.build(upper_left, _W1), None],
            [case.build(lower_left, _W1, _W2), case.build(lower_right, _W2)],
        ]
        op = BlockAssembled(grid, _ArrayPartition(case, _W, [_W1, _W2]))
        dense = np.block([[upper_left, np.zeros((3, 2))], [lower_left, lower_right]])
        return op, dense

    def test_action_and_format_equal_the_block_matrix(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """A zero block keeps its shape in the assembled format, and the
        stack's own direct solver factors it."""
        op, dense = self._blocks(case)
        expected = case.dense(dense)
        x = case.vec(_W, 2, 23)
        case.assert_vector(op.apply(x), case.times(expected, x), rtol=1e-12)
        case.assert_vector(op.apply_transpose(x), case.times(expected.T, x), rtol=1e-12)
        assert isinstance(op, ActionOperatorProtocol)
        assert isinstance(op.format(), case.format_type)
        assert op.format() is op.format()
        case.bkd.assert_allclose(case.format_dense(op.format()), expected, rtol=1e-12)
        case.assert_dot_identity(op, 24)
        solution, info = case.direct_solver().setup(op).solve(op.apply(x))
        assert info.converged
        case.assert_vector(solution, case.array(x), rtol=1e-10)

    def test_block_diagonal_keeps_properties(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        grid: list[list[Optional[ActionOperatorProtocol[V, F]]]] = [
            [case.identity(_W1), None],
            [None, case.build(spd_matrix(2, 25), _W2, properties=_SPD)],
        ]
        op = BlockAssembled(grid, _ArrayPartition(case, _W, [_W1, _W2]))
        assert op.properties() == OperatorProperties(symmetric=True, spd=True)

    def test_coupling_declares_only_the_transpose(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        op, _ = self._blocks(case)
        assert op.properties() == OperatorProperties()

    def test_rejects_a_misshapen_grid(self, case: TypedStackCase[V, F, A]) -> None:
        partition = _ArrayPartition(case, _W, [_W1, _W2])
        with pytest.raises(ValueError, match="2x2 grid"):
            BlockAssembled([[case.identity(_W1)]], partition)
        with pytest.raises(ValueError, match="at least one present block"):
            BlockAssembled([[None, None], [None, None]], partition)
        with pytest.raises(SpaceMismatchError):
            BlockAssembled(
                [[case.identity(_W2), None], [None, case.identity(_W2)]], partition
            )

    def test_mixed_formats_apply_but_have_no_format(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        stored = case.identity(_W1)
        free = as_matrix_free(case.identity(_W2))
        blocks: list[list[Optional[ActionOperatorProtocol[V, FormatProtocol]]]] = [
            [stored, None],
            [None, free],
        ]
        op = BlockAssembled(blocks, _ArrayPartition(case, _W, [_W1, _W2]))
        x = case.vec(_W, 1, 26)
        case.assert_vector(op.apply(x), case.array(x), rtol=1e-12)
        with pytest.raises(TypeError, match="different formats"):
            op.format()


class TransposedSuite:
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

    def test_refuses_an_operator_without_a_transpose(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        forward = ForwardOnly(case.build(random_matrix(3, 5, 9), _U, _P))
        with pytest.raises(ValueError, match="has no transpose"):
            Transposed(forward)


class TransposeCapabilitySuite:
    def test_combinators_carry_a_missing_transpose(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """A sum or product with a forward-only operand is forward-only,
        and still applies."""
        a = random_matrix(5, 5, 22)
        stored = case.build(a, _U)
        forward: ForwardOnly[V] = ForwardOnly(stored)
        total = ScaledSum(
            [(1.0, as_matrix_free(stored)), (2.0, forward)], case.space(_U)
        )
        factors: list[LinearMapProtocol[V]] = [stored, forward]
        product = Product(factors)
        assert not total.properties().transposable
        assert not product.properties().transposable
        assert ScaledSum([(1.0, stored)], case.space(_U)).properties().transposable
        x = case.vec(_U, 1, 23)
        case.assert_vector(total.apply(x), case.times(3.0 * case.dense(a), x))
        with pytest.raises(TypeError, match="no transpose"):
            total.apply_transpose(x)


class ProductSuite:
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


class AsMatrixFreeSuite:
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
