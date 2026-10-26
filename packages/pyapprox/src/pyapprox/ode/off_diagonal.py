r"""The off-diagonal block of a one-step time-stepping Jacobian, as an operator.

A one-step method's residual :math:`R_n(y_n, y_{n-1})` couples step
:math:`n` to step :math:`n-1` through
:math:`B_n = \partial R_n/\partial y_{n-1}`. The forward-sensitivity
sweep applies :math:`B_n`; the adjoint sweep applies :math:`B_n^\top`.
The operators here apply both without forming :math:`B_n`, from the
residual's mass and Jacobian actions, so a matrix-free residual or a
large sparse one never needs the block assembled.

Each operator is ``MatrixFree`` on blocks of shape ``(nstates, ncols)``
and refers to the states it was made at rather than copying them.

The Jacobian actions come from the residual's matrix ``jacobian`` for
now (dense backend arrays, or SciPy sparse matrices from Galerkin); they
are re-sourced from the residual's derivatives bundle once every
residual reaching the sweeps provides one, without changing the
formulas.
"""

from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Union

from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import (
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)


class _MatrixAction(Generic[Array]):
    """A Jacobian matrix applied to one vector and transposed, whether it
    is a dense backend array or a SciPy sparse matrix (Galerkin), until
    the Jacobian comes from the residual's bundle as an operator."""

    def __init__(self, matrix: Union[spmatrix, Array], bkd: Backend[Array]) -> None:
        self._matrix, self._bkd = matrix, bkd

    def apply(self, vec: Array) -> Array:
        if isinstance(self._matrix, spmatrix):
            return self._bkd.asarray(self._matrix @ self._bkd.to_numpy(vec))
        return self._bkd.dot(self._matrix, vec)

    def apply_transpose(self, vec: Array) -> Array:
        if isinstance(self._matrix, spmatrix):
            return self._bkd.asarray(self._matrix.T @ self._bkd.to_numpy(vec))
        return self._bkd.dot(self._matrix.T, vec)


class _OffDiagonalOperator(ABC, Generic[Array]):
    """What the off-diagonal operators share: the state space, the
    ``MatrixFree`` format, and applying a vector action column by
    column."""

    def __init__(self, nstates: int, bkd: Backend[Array]) -> None:
        self._descriptor = SpaceDescriptor("state", nstates, backend_dtype_name(bkd))
        self._bkd = bkd

    def domain(self) -> SpaceDescriptor:
        return self._descriptor

    def range(self) -> SpaceDescriptor:
        return self._descriptor

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    @abstractmethod
    def _apply_vector(self, vec: Array) -> Array:
        r""":math:`B_n v` for one vector. Shape: (nstates,)"""
        ...

    @abstractmethod
    def _apply_transpose_vector(self, vec: Array) -> Array:
        r""":math:`B_n^\top w` for one vector. Shape: (nstates,)"""
        ...

    def _columns(self, action: Callable[[Array], Array], block: Array) -> Array:
        size = self._descriptor.size
        if block.ndim != 2 or block.shape[0] != size:
            raise ValueError(
                f"input must have shape ({size}, ncols), got {tuple(block.shape)}"
            )
        return self._bkd.stack(
            [action(block[:, k]) for k in range(block.shape[1])], axis=1
        )

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._apply_vector, x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._columns(self._apply_transpose_vector, y)


class MassPlusJacobianOffDiagonal(_OffDiagonalOperator[Array]):
    r""":math:`B_n = -(M + c\,J)`, the block of every one-step method whose
    previous state enters through the mass and one Jacobian evaluation:
    backward Euler (:math:`c = 0`), forward Euler (:math:`c = \Delta t`,
    :math:`J` at :math:`y_{n-1}`), Crank-Nicolson (:math:`c = \Delta t/2`,
    :math:`J` at :math:`y_{n-1}`) and implicit midpoint
    (:math:`c = \Delta t/2`, :math:`J` at the midpoint).

    Parameters
    ----------
    mass : MassMatrixProtocol
        :math:`M`.
    jacobian : matrix or None
        :math:`J` at the method's point, or None when :math:`c = 0`.
    coefficient : float
        :math:`c`.
    nstates : int
        The number of states.
    bkd : Backend
        The backend of the states.
    """

    def __init__(
        self,
        mass: MassMatrixProtocol[Array],
        jacobian: Optional[Union[spmatrix, Array]],
        coefficient: float,
        nstates: int,
        bkd: Backend[Array],
    ) -> None:
        super().__init__(nstates, bkd)
        self._mass = mass
        self._jacobian = None if jacobian is None else _MatrixAction(jacobian, bkd)
        self._coefficient = coefficient

    def _apply_vector(self, vec: Array) -> Array:
        result = self._mass.apply(vec)
        if self._jacobian is not None:
            result = result + self._coefficient * self._jacobian.apply(vec)
        return -result

    def _apply_transpose_vector(self, vec: Array) -> Array:
        result = self._mass.apply_transpose(vec)
        if self._jacobian is not None:
            result = result + self._coefficient * self._jacobian.apply_transpose(vec)
        return -result


class HeunOffDiagonal(_OffDiagonalOperator[Array]):
    r"""Heun's block :math:`B_n = -\bigl(M + \tfrac{\Delta t}{2}
    (J_1 + J_2 (I + \Delta t\,M^{-1} J_1))\bigr)`.

    :math:`J_1 = J(y_{n-1}, t_{n-1})` and :math:`J_2 = J(z, t_n)` at the
    predictor :math:`z = y_{n-1} + \Delta t\,M^{-1} f(y_{n-1}, t_{n-1})`.
    Applied as actions, so :math:`M^{-1} J_1` is never formed:
    :math:`B_n v = -(M v + \tfrac{\Delta t}{2}(u + J_2(v + \Delta t\,
    M^{-1} u)))` with :math:`u = J_1 v`, and :math:`B_n^\top w =
    -(M^\top w + \tfrac{\Delta t}{2}(J_1^\top w + a + \Delta t\,J_1^\top
    M^{-\top} a))` with :math:`a = J_2^\top w`.

    Parameters
    ----------
    mass : MassMatrixProtocol
        :math:`M`, solved with for the predictor slope.
    jacobian_previous, jacobian_predictor : matrix
        :math:`J_1` and :math:`J_2`.
    deltat : float
        :math:`\Delta t`.
    nstates : int
        The number of states.
    bkd : Backend
        The backend of the states.
    """

    def __init__(
        self,
        mass: MassMatrixProtocol[Array],
        jacobian_previous: Union[spmatrix, Array],
        jacobian_predictor: Union[spmatrix, Array],
        deltat: float,
        nstates: int,
        bkd: Backend[Array],
    ) -> None:
        super().__init__(nstates, bkd)
        self._mass = mass
        self._j1 = _MatrixAction(jacobian_previous, bkd)
        self._j2 = _MatrixAction(jacobian_predictor, bkd)
        self._deltat = deltat

    def _apply_vector(self, vec: Array) -> Array:
        dt = self._deltat
        u = self._j1.apply(vec)
        stage = vec + dt * self._mass.solve(u)
        total: Array = self._mass.apply(vec) + 0.5 * dt * (u + self._j2.apply(stage))
        return -total

    def _apply_transpose_vector(self, vec: Array) -> Array:
        dt = self._deltat
        a = self._j2.apply_transpose(vec)
        coupled = (
            self._j1.apply_transpose(vec)
            + a
            + dt * self._j1.apply_transpose(self._mass.solve_transpose(a))
        )
        total: Array = self._mass.apply_transpose(vec) + 0.5 * dt * coupled
        return -total
