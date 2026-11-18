"""Stack-generic operator suites: ``LinearizedOperatorSuite``.

Each runs on the stack its ``case`` fixture supplies
(``TypedStackCase``); a stack's test directory subclasses it and
defines the fixture.
"""

from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    JacobianSnapshotProtocol,
    LinearizationPoint,
    LinearizedOperator,
    OperatorProperties,
    SpaceDescriptor,
)
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


class LinearizedOperatorSuite:
    def test_carries_its_point_and_delegates(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        a = random_matrix(3, 5, 0)
        stored = case.build(a, _U, _P)
        point = LinearizationPoint(1, "iterate-2", provisional=True)
        snapshot = LinearizedOperator(stored, point)
        assert isinstance(snapshot, JacobianSnapshotProtocol)
        assert isinstance(snapshot, ActionOperatorProtocol)
        assert snapshot.linearization() is point
        assert snapshot.domain() == _U and snapshot.range() == _P
        assert snapshot.format() is stored.format()
        x, y = case.vec(_U, 2, 1), case.vec(_P, 2, 2)
        case.assert_vector(snapshot.apply(x), case.array(stored.apply(x)))
        case.assert_vector(
            snapshot.apply_transpose(y), case.array(stored.apply_transpose(y))
        )

    def test_keeps_properties_and_solves(self, case: TypedStackCase[V, F, A]) -> None:
        """A labeled derivative is still factorable in its format."""
        stored = case.build(spd_matrix(5, 3), _U, properties=_SPD)
        snapshot = LinearizedOperator(stored, LinearizationPoint(0))
        assert snapshot.properties() == _SPD
        b = case.vec(_U, 1, 4)
        x, info = case.direct_solver().setup(snapshot).solve(b)
        assert info.converged
        case.assert_vector(stored.apply(x), case.array(b), rtol=1e-10)

    def test_a_plain_operator_is_not_a_snapshot(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        assert not isinstance(case.identity(_U), JacobianSnapshotProtocol)
