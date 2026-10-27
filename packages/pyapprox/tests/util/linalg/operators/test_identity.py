"""IdentityOperator on every stack."""

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    IdentityOperator,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)
from tests._helpers.linalg.stacks import A, F, TypedStackCase, V

_U = SpaceDescriptor("u", 5)


class TestIdentityOperator:
    def test_declares_the_identity(self, case: TypedStackCase[V, F, A]) -> None:
        op = IdentityOperator(case.space(_U))
        assert isinstance(op, ActionOperatorProtocol)
        assert op.domain() == _U and op.range() == _U
        assert op.properties() == OperatorProperties(
            symmetric=True, spd=True, diagonal=True, identity=True
        )
        assert isinstance(op.format(), MatrixFree)

    def test_returns_a_copy_of_its_input(self, case: TypedStackCase[V, F, A]) -> None:
        """Exactly equal values in a new block, both ways, so a caller may
        modify the result without touching the input."""
        op = IdentityOperator(case.space(_U))
        x = case.vec(_U, 2, 0)
        expected = case.array(x)
        for result in (op.apply(x), op.apply_transpose(x)):
            assert result is not x
            case.assert_vector(result, expected, rtol=0.0)
