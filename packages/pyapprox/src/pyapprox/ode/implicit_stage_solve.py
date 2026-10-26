r"""Linear algebra of an implicit stage, through a residual-form system.

A stage of an implicit one-step method has Newton matrix
:math:`dR/dy = M - cJ`, which is :math:`c(\sigma M - J)` with
:math:`\sigma = 1/c`: the shifted Jacobian of an
:class:`~pyapprox.ode.protocols.ResidualFormSystemProtocol` scaled by
:math:`c`. ``ImplicitStageSolve`` turns a system into the operators a
stepper solves with, so the stepper holds one and needs no knowledge of
the system's stack or format.

The stepper-facing result is the matrix-era ``LinearOperatorProtocol``
(``solve``, ``solve_transpose``, ``apply``, ``as_matrix``), because the
time-step sweeps still consume it; ``as_matrix`` is the one call that
forms a matrix, from a function the stepper passes.
"""

from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.ode.linear_operator import LinearOperatorProtocol
from pyapprox.ode.mass_matrix_system import dense_system
from pyapprox.ode.protocols import ODEResidualProtocol, ResidualFormSystemProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
    SolveFailedError,
    SolverHandleProtocol,
)

F = TypeVar("F")


def newton_matrix(
    residual: ODEResidualProtocol[Array], state: Array, time: float, coefficient: float
) -> Array:
    """:math:`M - cJ(y, t)` as a matrix, from the residual's mass and its
    matrix Jacobian, in the mass's own storage (a diagonal mass adds only
    its diagonal, so a sparse Jacobian stays sparse)."""
    return residual.mass_matrix().add_to(-coefficient * residual.jacobian(state, time))


@runtime_checkable
class ImplicitSystemProviderProtocol(Protocol[Array, F]):
    """A model that supplies its own residual-form system."""

    def implicit_system(self) -> ResidualFormSystemProtocol[Array, Array, F]:
        """The model's residual-form system, in its natural stack and format."""
        ...


class _ScaledShiftedOperator(Generic[Array, F]):
    r""":math:`c(\sigma M - J)` at one linearization, solved by a handle
    of the system's solver, set up on first solve.

    The system's shifted Jacobian is built on first solve or action, not
    at construction, so a caller that only asks for ``as_matrix`` never
    assembles it. Like the shifted Jacobian, it reads ``state`` and
    ``state_rate`` then, so they must be left unmodified until it is
    used.
    """

    def __init__(
        self,
        system: ResidualFormSystemProtocol[Array, Array, F],
        state: Array,
        state_rate: Array,
        time: float,
        coefficient: float,
        point: LinearizationPoint,
        matrix: Callable[[], Array],
        bkd: Backend[Array],
    ) -> None:
        self._system = system
        self._coefficient = coefficient
        self._linearization = (state, state_rate, time, point)
        self._operator: Optional[JacobianSnapshotProtocol[Array, F]] = None
        self._matrix = matrix
        self._bkd = bkd
        self._handle: Optional[SolverHandleProtocol[Array]] = None

    def _shifted(self) -> JacobianSnapshotProtocol[Array, F]:
        if self._operator is None:
            state, state_rate, time, point = self._linearization
            self._operator = self._system.shifted_jacobian(
                state, state_rate, time, 1.0 / self._coefficient, point
            )
        return self._operator

    def _solve_with(self, handle: SolverHandleProtocol[Array], rhs: Array) -> Array:
        column = rhs.ndim == 1
        block = self._bkd.reshape(rhs, (rhs.shape[0], 1)) if column else rhs
        solution, info = handle.solve(block)
        if not info.converged:
            raise SolveFailedError(info.message)
        result = solution / self._coefficient
        return self._bkd.reshape(result, (rhs.shape[0],)) if column else result

    def _setup(self) -> SolverHandleProtocol[Array]:
        if self._handle is None:
            self._handle = self._system.solver().setup(self._shifted())
        return self._handle

    def solve(self, rhs: Array) -> Array:
        """Solve :math:`(dR/dy) x = b`."""
        return self._solve_with(self._setup(), rhs)

    def solve_transpose(self, rhs: Array) -> Array:
        r"""Solve :math:`(dR/dy)^\top x = b`."""
        transposed = self._setup().transpose()
        if transposed is None:
            raise TypeError(
                f"{type(self._system.solver()).__name__} has no transposed solve"
            )
        return self._solve_with(transposed, rhs)

    def _apply_with(self, action: Callable[[Array], Array], vec: Array) -> Array:
        column = vec.ndim == 1
        block = self._bkd.reshape(vec, (vec.shape[0], 1)) if column else vec
        result = self._coefficient * action(block)
        return self._bkd.reshape(result, (vec.shape[0],)) if column else result

    def apply(self, vec: Array) -> Array:
        return self._apply_with(self._shifted().apply, vec)

    def apply_transpose(self, vec: Array) -> Array:
        return self._apply_with(self._shifted().apply_transpose, vec)

    def as_matrix(self) -> Array:
        """:math:`dR/dy` as a matrix, formed by the stepper's function."""
        return self._matrix()


class _SystemLinearizer(Generic[Array, F]):
    """Binds a system, so each linearization pairs its shifted Jacobian
    with a solver of the same format; the stepper sees neither ``F``."""

    def __init__(self, system: ResidualFormSystemProtocol[Array, Array, F]) -> None:
        if not isinstance(system, ResidualFormSystemProtocol):
            raise TypeError(
                "system must satisfy ResidualFormSystemProtocol, "
                f"got {type(system).__name__}"
            )
        self._system = system

    def newton_operator(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        coefficient: float,
        point: LinearizationPoint,
        matrix: Callable[[], Array],
        bkd: Backend[Array],
    ) -> LinearOperatorProtocol[Array]:
        return _ScaledShiftedOperator(
            self._system, state, state_rate, time, coefficient, point, matrix, bkd
        )


class _LinearizerProtocol(Protocol[Array]):
    def newton_operator(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        coefficient: float,
        point: LinearizationPoint,
        matrix: Callable[[], Array],
        bkd: Backend[Array],
    ) -> LinearOperatorProtocol[Array]: ...


class ImplicitStageSolve(Generic[Array]):
    r"""The Newton operators of an implicit stepper's stages.

    Parameters
    ----------
    residual : ODEResidualProtocol
        The stepper's residual; it supplies the system when ``system``
        is not given.
    system : ResidualFormSystemProtocol, optional
        The residual-form system to solve with. When omitted, the residual's
        own ``implicit_system()`` is used if it has one, and otherwise
        ``dense_system(residual)``; the choice is made on first use, so
        constructing a stepper costs nothing.
    """

    def __init__(
        self,
        residual: ODEResidualProtocol[Array],
        system: Optional[ResidualFormSystemProtocol[Array, Array, F]] = None,
    ) -> None:
        self._residual = residual
        self._bkd = residual.bkd()
        self._linearizer: Optional[_LinearizerProtocol[Array]] = (
            None if system is None else _SystemLinearizer(system)
        )
        self._linearizations = 0

    def stage_point(self, time: float) -> LinearizationPoint:
        """A label for one linearization of a stage at ``time``.

        Each call returns a new label. A caching solver reuses a setup
        only for an equal label, so implicit stages are never reused:
        not across Newton iterates, nor across parameter values.
        """
        self._linearizations += 1
        return LinearizationPoint(
            0, ("implicit stage", time, self._linearizations), provisional=True
        )

    def _resolve(self) -> _LinearizerProtocol[Array]:
        if self._linearizer is None:
            residual = self._residual
            if isinstance(residual, ImplicitSystemProviderProtocol):
                self._linearizer = _SystemLinearizer(residual.implicit_system())
            else:
                self._linearizer = _SystemLinearizer(dense_system(residual))
        return self._linearizer

    def newton_operator(
        self,
        state: Array,
        state_rate: Array,
        time: float,
        coefficient: float,
        point: LinearizationPoint,
        matrix: Callable[[], Array],
    ) -> LinearOperatorProtocol[Array]:
        r""":math:`dR/dy = c(\sigma M - J)` linearized at ``(state, time)``.

        Parameters
        ----------
        state, state_rate, time
            Where the system is linearized; ``state_rate`` is the stage's
            :math:`\dot y` approximation.
        coefficient : float
            :math:`c`, so the shift is :math:`1/c`.
        point : LinearizationPoint
            The identity of the linearization.
        matrix : callable
            Forms :math:`dR/dy` as a matrix, for ``as_matrix``.
        """
        return self._resolve().newton_operator(
            state, state_rate, time, coefficient, point, matrix, self._bkd
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(residual={self._residual!r})"
