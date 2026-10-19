"""An operator-valued Jacobian from a model that applies its Jacobian.

``MatrixFreeJacobianOperator`` satisfies
:class:`~pyapprox.ode.protocols.JacobianOperatorProtocol` for a model
that supplies the actions :math:`J v` and :math:`J^T w` instead of
the matrix. Each result is a :class:`MatrixFreeJacobianSnapshot` in the
:class:`~pyapprox.util.linalg.operators.MatrixFree` format, so it pairs
with an iterative solver, never a direct one. It is worth it when the
Jacobian is expensive to form or store and its actions are cheap; for a
small system the matrix and a dense solve are faster.

The model's actions see whole backend arrays, so this is a
single-process tool. Applying a Jacobian of a distributed state needs
ghost values near partition boundaries, which element-wise kernels get
from their assembler rather than from model code.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import (
    LinearizationPoint,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)


@runtime_checkable
class JacobianActionProtocol(Protocol[Array]):
    """A model that applies its state Jacobian and its transpose.

    Both actions are required: Newton needs :math:`J v`, adjoint
    gradients need :math:`J^T w`. Each takes and returns blocks of
    shape (nstates, ncols) and returns a new array.
    """

    def apply_jacobian(self, state: Array, time: float, v: Array) -> Array:
        """:math:`J(state, time) v`."""
        ...

    def apply_jacobian_transpose(self, state: Array, time: float, w: Array) -> Array:
        """:math:`J(state, time)^T w`."""
        ...


class MatrixFreeJacobianSnapshot(Generic[Array]):
    """The Jacobian at one point, applied through the model.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.JacobianSnapshotProtocol`
    with backend arrays as vectors. It refers to ``state`` rather than
    copying it, so it is valid while the caller leaves ``state``
    unmodified. ``out`` is ignored: the model returns new arrays.

    Parameters
    ----------
    model : JacobianActionProtocol
        Supplies the actions.
    state : Array
        Where the Jacobian is taken. Shape: (nstates,)
    time : float
        Time t.
    descriptor : SpaceDescriptor
        The state space: domain and range.
    point : LinearizationPoint
        The identity of ``(state, time)``.
    """

    def __init__(
        self,
        model: JacobianActionProtocol[Array],
        state: Array,
        time: float,
        descriptor: SpaceDescriptor,
        point: LinearizationPoint,
    ) -> None:
        self._model = model
        self._state = state
        self._time = time
        self._descriptor = descriptor
        self._point = point

    def domain(self) -> SpaceDescriptor:
        return self._descriptor

    def range(self) -> SpaceDescriptor:
        return self._descriptor

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def linearization(self) -> LinearizationPoint:
        """Where the Jacobian was taken."""
        return self._point

    def _check(self, block: Array, name: str) -> None:
        size = self._descriptor.size
        if block.ndim != 2 or block.shape[0] != size:
            raise ValueError(
                f"{name} must be a block of shape ({size}, ncols), "
                f"got {tuple(block.shape)}"
            )

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        self._check(x, "x")
        result = self._model.apply_jacobian(self._state, self._time, x)
        self._check_result(result, x)
        return result

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        self._check(y, "y")
        result = self._model.apply_jacobian_transpose(self._state, self._time, y)
        self._check_result(result, y)
        return result

    def _check_result(self, result: Array, block: Array) -> None:
        if tuple(result.shape) != tuple(block.shape):
            raise ValueError(
                f"the model's action returned shape {tuple(result.shape)} "
                f"for an input of shape {tuple(block.shape)}"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._descriptor!r}, {self._point!r})"


class MatrixFreeJacobianOperator(Generic[Array]):
    """The Jacobian of a model as labeled matrix-free operators.

    Parameters
    ----------
    model : JacobianActionProtocol
        Supplies :math:`J v` and :math:`J^T w`.
    descriptor : SpaceDescriptor
        The state space: the operators' domain and range.

    Raises
    ------
    TypeError
        If ``model`` does not satisfy ``JacobianActionProtocol``.
    """

    def __init__(
        self, model: JacobianActionProtocol[Array], descriptor: SpaceDescriptor
    ) -> None:
        if not isinstance(model, JacobianActionProtocol):
            raise TypeError(
                f"model must satisfy JacobianActionProtocol, got {type(model).__name__}"
            )
        self._model = model
        self._descriptor = descriptor

    def jacobian_operator(
        self, state: Array, time: float, point: LinearizationPoint
    ) -> MatrixFreeJacobianSnapshot[Array]:
        """The Jacobian at ``(state, time)``, labeled with ``point``; it
        refers to ``state``, which the caller must leave unmodified while
        the result is in use."""
        return MatrixFreeJacobianSnapshot(
            self._model, state, time, self._descriptor, point
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
