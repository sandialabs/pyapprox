"""ZeroOperator on every stack, and the ``zero`` operator property."""

import pytest

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
    ZeroOperator,
)
from tests._helpers.linalg.stacks import A, F, TypedStackCase, V

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)


class TestZeroOperator:
    def test_maps_any_block_to_zeros_between_two_spaces(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        op = ZeroOperator(case.space(_P), case.space(_U))
        assert isinstance(op, ActionOperatorProtocol)
        assert op.domain() == _P and op.range() == _U
        assert op.properties().zero and isinstance(op.format(), MatrixFree)
        case.assert_vector(op.apply(case.vec(_P, 2, 0)), case.bkd.zeros((5, 2)))
        case.assert_vector(
            op.apply_transpose(case.vec(_U, 3, 1)), case.bkd.zeros((3, 3))
        )

    def test_defaults_to_a_square_operator(self, case: TypedStackCase[V, F, A]) -> None:
        op = ZeroOperator(case.space(_U))
        assert op.domain() == _U and op.range() == _U


class TestZeroProperty:
    def test_excludes_identity(self) -> None:
        assert OperatorProperties(zero=True).zero
        with pytest.raises(ValueError, match="excludes identity"):
            OperatorProperties(
                symmetric=True, spd=True, diagonal=True, identity=True, zero=True
            )
