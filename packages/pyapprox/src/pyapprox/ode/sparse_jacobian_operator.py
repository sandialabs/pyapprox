"""An operator-valued Jacobian from a model that returns a sparse matrix.

``SparseJacobianOperator`` satisfies
:class:`~pyapprox.ode.protocols.JacobianOperatorProtocol` on the
NumPy/SciPy stack for a model whose ``jacobian(state, time)`` returns a
SciPy sparse matrix, such as an assembled finite-element discretization.
Each result is a :class:`SparseJacobianSnapshot`, a CSR operator that
carries its own linearization point, so it pairs with a sparse direct
solver. It is bound to NumPy and SciPy, so it is not differentiable.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.ode.dense_jacobian_operator import JacobianMatrixProtocol
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator

# NumPy backend arrays: the dtype varies with the model.
_Np = NDArray[Any]


class SparseJacobianSnapshot(ScipyCSROperator[np.float64]):
    """A CSR Jacobian operator that records where it was taken.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.JacobianSnapshotProtocol`.

    Parameters
    ----------
    matrix : csr_matrix
        The Jacobian. Shape: (descriptor.size, descriptor.size)
    descriptor : SpaceDescriptor
        The state space: domain and range; float64.
    point : LinearizationPoint
        Where the Jacobian was taken.
    """

    def __init__(
        self, matrix: csr_matrix, descriptor: SpaceDescriptor, point: LinearizationPoint
    ) -> None:
        super().__init__(matrix, descriptor, scalar=np.float64)
        self._point = point

    def linearization(self) -> LinearizationPoint:
        """Where the Jacobian was taken."""
        return self._point


class SparseJacobianOperator:
    """The sparse Jacobian of a model as labeled CSR operators.

    Parameters
    ----------
    model : JacobianMatrixProtocol
        Supplies ``jacobian(state, time)`` as a SciPy sparse matrix (or
        anything ``csr_matrix`` accepts).
    descriptor : SpaceDescriptor
        The state space; float64.

    Raises
    ------
    TypeError
        If ``model`` does not satisfy ``JacobianMatrixProtocol``.
    """

    def __init__(
        self, model: JacobianMatrixProtocol[_Np], descriptor: SpaceDescriptor
    ) -> None:
        if not isinstance(model, JacobianMatrixProtocol):
            raise TypeError(
                f"model must satisfy JacobianMatrixProtocol, got {type(model).__name__}"
            )
        self._model = model
        self._descriptor = descriptor

    def jacobian_operator(
        self, state: _Np, time: float, point: LinearizationPoint
    ) -> SparseJacobianSnapshot:
        """The Jacobian at ``(state, time)``, labeled with ``point``.

        Raises
        ------
        ValueError
            If the model's Jacobian does not have the descriptor's shape.
        """
        return SparseJacobianSnapshot(
            csr_matrix(self._model.jacobian(state, time)), self._descriptor, point
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"
