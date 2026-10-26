r"""A residual-form system backed by a residual's own Newton matrix.

``NewtonJacobianSystem`` satisfies
:class:`~pyapprox.ode.protocols.ResidualFormSystemProtocol` for a residual
that builds its Newton operator :math:`M - cJ` itself, through
``newton_jacobian(state, time, coefficient)``: the shifted Jacobian is
:math:`\sigma M - J = \sigma\,(M - J/\sigma)`, applied and solved with
that operator, in whatever storage the residual uses. Its shifted
Jacobian is ``MatrixFree`` and pairs only with its own solver.

It adapts residuals that already build their own Newton operator and
have no stack-typed system; it is not a way to define new models.
New models give their mass and Jacobian to a ``MassMatrixSystem``
(see ``dense_system``, ``sparse_system``, ``matrix_free_system``).
"""

from typing import Generic, Optional

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.protocols.ode_residual import ImplicitODEResidualProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import backend_dtype_name
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearizationPoint,
    MatrixFree,
    OperatorProperties,
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
    SpaceDescriptor,
)


class NewtonJacobianSnapshot(Generic[Array]):
    r""":math:`\sigma M - J` at one linearization, through the residual's
    Newton operator for :math:`c = 1/\sigma`.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.JacobianSnapshotProtocol`.
    """

    def __init__(
        self,
        newton: LinearOperatorProtocol[Array],
        shift: float,
        descriptor: SpaceDescriptor,
        point: LinearizationPoint,
    ) -> None:
        self._newton = newton
        self._shift = shift
        self._descriptor = descriptor
        self._point = point

    def newton(self) -> LinearOperatorProtocol[Array]:
        r"""The residual's :math:`M - J/\sigma`."""
        return self._newton

    def shift(self) -> float:
        r""":math:`\sigma`."""
        return self._shift

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

    def apply(self, x: Array, out: Optional[Array] = None) -> Array:
        return self._shift * self._newton.apply(x)

    def apply_transpose(self, y: Array, out: Optional[Array] = None) -> Array:
        return self._shift * self._newton.apply_transpose(y)


class _NewtonJacobianHandle(SolverHandleBase[Array]):
    r"""Solves with :math:`\sigma M - J` by the Newton operator's own
    solve, or its transposed solve."""

    def __init__(
        self, snapshot: NewtonJacobianSnapshot[Array], transposed: bool
    ) -> None:
        super().__init__(snapshot)
        self._snapshot = snapshot
        self._transposed = transposed

    def solve(
        self, rhs: Array, tol: Optional[float] = None, x0: Optional[Array] = None
    ) -> tuple[Array, SolveInfo]:
        newton = self._snapshot.newton()
        solution = (
            newton.solve_transpose(rhs) if self._transposed else newton.solve(rhs)
        )
        return solution / self._snapshot.shift(), SolveInfo(True)

    def transpose(self) -> "_NewtonJacobianHandle[Array]":
        return _NewtonJacobianHandle(self._snapshot, not self._transposed)


class NewtonJacobianSolver(Generic[Array]):
    """The solver of a ``NewtonJacobianSystem``'s shifted Jacobians.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for ``MatrixFree``, but only for the system's own snapshots, whose
    Newton operator it solves with.
    """

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(
        self, op: ActionOperatorProtocol[Array, MatrixFree]
    ) -> SolverHandleProtocol[Array]:
        """Raises ``TypeError`` for an operator that is not a
        ``NewtonJacobianSnapshot``."""
        if not isinstance(op, NewtonJacobianSnapshot):
            raise TypeError(
                "NewtonJacobianSolver solves only the shifted Jacobians of a "
                f"NewtonJacobianSystem, got {type(op).__name__}"
            )
        return _NewtonJacobianHandle(op, transposed=False)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NewtonJacobianSystem(Generic[Array]):
    r""":math:`F = M \dot y - f(y, t)` with the residual's own Newton matrix.

    Satisfies :class:`~pyapprox.ode.protocols.ResidualFormSystemProtocol`
    with ``MatrixFree`` shifted Jacobians.

    Parameters
    ----------
    residual : ImplicitODEResidualProtocol
        Supplies :math:`f`, its mass and ``newton_jacobian``.

    Raises
    ------
    TypeError
        If ``residual`` has no ``newton_jacobian``.
    """

    def __init__(self, residual: ImplicitODEResidualProtocol[Array]) -> None:
        if not isinstance(residual, ImplicitODEResidualProtocol):
            raise TypeError(
                "residual must satisfy ImplicitODEResidualProtocol, "
                f"got {type(residual).__name__}"
            )
        self._residual = residual
        self._bkd: Backend[Array] = residual.bkd()
        nstates = residual.mass_matrix().as_matrix().shape[0]
        self._descriptor = SpaceDescriptor(
            "state", nstates, backend_dtype_name(self._bkd)
        )

    def residual(self, state: Array, state_rate: Array, time: float) -> Array:
        r""":math:`M \dot y - f(y, t)`. Shape: (nstates,)"""
        return self._residual.mass_matrix().apply(state_rate) - self._residual(
            state, time
        )

    def shifted_jacobian(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        shift: float,
        point: LinearizationPoint,
    ) -> NewtonJacobianSnapshot[Array]:
        r""":math:`\sigma M - J(y, t)`, labeled with ``point``."""
        newton = self._residual.newton_jacobian(state, time, 1.0 / shift)
        return NewtonJacobianSnapshot(newton, shift, self._descriptor, point)

    def solver(self) -> NewtonJacobianSolver[Array]:
        return NewtonJacobianSolver()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(residual={self._residual!r})"
