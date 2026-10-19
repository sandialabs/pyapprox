"""An operator-valued Jacobian from a model that returns its Jacobian matrix.

``DenseJacobianOperator`` satisfies
:class:`~pyapprox.ode.protocols.JacobianOperatorProtocol` for any model
with ``jacobian(state, time)`` returning a dense backend array, such as a
small ODE or a collocation discretization. Each result is a
:class:`DenseJacobianSnapshot`, a dense operator of the backend stack
that carries its own linearization point, so it pairs with a dense
direct solver. Everything is backend arithmetic, so on a differentiating
backend the gradient flows through the operator's action.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor


@runtime_checkable
class JacobianMatrixProtocol(Protocol[Array]):
    """A model that returns its state Jacobian as a dense backend array."""

    def jacobian(self, state: Array, time: float) -> Array:
        """``df/dy`` at ``(state, time)``. Shape: (nstates, nstates)."""
        ...


class DenseJacobianSnapshot(BackendDenseOperator[Array]):
    """A dense Jacobian operator that records where it was taken.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.JacobianSnapshotProtocol`.

    Parameters
    ----------
    matrix : Array
        The Jacobian. Shape: (descriptor.size, descriptor.size)
    descriptor : SpaceDescriptor
        The state space: domain and range.
    bkd : Backend
        Computational backend.
    point : LinearizationPoint
        Where the Jacobian was taken.
    """

    def __init__(
        self,
        matrix: Array,
        descriptor: SpaceDescriptor,
        bkd: Backend[Array],
        point: LinearizationPoint,
    ) -> None:
        super().__init__(matrix, descriptor, bkd)
        self._point = point

    def linearization(self) -> LinearizationPoint:
        """Where the Jacobian was taken."""
        return self._point


class DenseJacobianOperator(Generic[Array]):
    """The Jacobian of a model as labeled dense operators.

    Parameters
    ----------
    model : JacobianMatrixProtocol
        Supplies ``jacobian(state, time)``.
    descriptor : SpaceDescriptor
        The state space: the operators' domain and range.
    bkd : Backend
        Computational backend.

    Raises
    ------
    TypeError
        If ``model`` does not satisfy ``JacobianMatrixProtocol``.
    """

    def __init__(
        self,
        model: JacobianMatrixProtocol[Array],
        descriptor: SpaceDescriptor,
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(model, JacobianMatrixProtocol):
            raise TypeError(
                f"model must satisfy JacobianMatrixProtocol, got {type(model).__name__}"
            )
        self._model = model
        self._descriptor = descriptor
        self._bkd = bkd

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> DenseJacobianSnapshot[Array]:
        """The Jacobian at ``(state, time)``, labeled with ``point``.

        Raises
        ------
        ValueError
            If the model's Jacobian does not have the descriptor's shape.
        """
        return DenseJacobianSnapshot(
            self._model.jacobian(state, time), self._descriptor, self._bkd, point
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
