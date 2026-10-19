"""JacobianOperatorProtocol: what a time stepper consumes.

A linear ODE :math:`f(y) = A y` on the backend-dense stack is the
smallest producer; a consumer typed on the protocol alone applies it, so
conformance is checked statically as well as at runtime.
"""

from typing import Generic

from pyapprox.ode.protocols import JacobianOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, DenseFormat
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    LinearizedOperator,
    SpaceDescriptor,
)

_U = SpaceDescriptor("y", 3)


class _LinearODE(Generic[Array]):
    """``f(y) = A y``: the Jacobian is ``A`` at every state."""

    def __init__(self, matrix: Array, bkd: Backend[Array]) -> None:
        self._matrix = matrix
        self._bkd = bkd

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> LinearizedOperator[Array, DenseFormat[Array]]:
        return LinearizedOperator(
            BackendDenseOperator(self._matrix, _U, self._bkd), point
        )


def _apply_jacobian(
    producer: JacobianOperatorProtocol[Array, Array, DenseFormat[Array]],
    state: Array,
    direction: Array,
    point: LinearizationPoint,
) -> Array:
    """A consumer that knows only the protocol."""
    op = producer.jacobian_operator(state, 0.0, point)
    assert op.linearization() == point
    return op.apply(direction)


class TestJacobianOperatorProtocol:
    def test_linear_ode_satisfies_and_applies(self, bkd: Backend[Array]) -> None:
        matrix = bkd.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
        producer = _LinearODE(matrix, bkd)
        assert isinstance(producer, JacobianOperatorProtocol)
        direction = bkd.asarray([[1.0], [-1.0], [2.0]])
        result = _apply_jacobian(
            producer, bkd.zeros((3,)), direction, LinearizationPoint(0, "y0")
        )
        bkd.assert_allclose(result, bkd.dot(matrix, direction))

    def test_an_object_without_the_method_is_refused(self) -> None:
        assert not isinstance(object(), JacobianOperatorProtocol)
