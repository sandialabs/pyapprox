r"""Linear solvers and their handles.

A *solver* is set up for one operator :math:`A` and returns a *handle*
that solves :math:`A x = b`. The handle is itself an operator, from
:math:`A`'s range to its domain, whose ``apply`` is the solve, so an
inverse composes anywhere an operator does: inside a product, as a block
of a block operator, or as another solver's preconditioner. There is no
separate preconditioner type: a preconditioner is a solver handle that
solves approximately.

``solve`` never raises on failure; it reports through
:class:`SolveInfo`, and callers must check ``converged``. ``apply``
cannot return that report, so it raises :class:`SolveFailedError`
instead.

A solver states the operator format it needs in its ``setup`` signature,
which mypy checks, and checks it again at ``setup``, raising
``TypeError`` for a pairing chosen by runtime data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import MatrixFree
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    DiagonalCapableProtocol,
    OperatorProperties,
    VectorSpaceProtocol,
)

V = TypeVar("V")
F_contra = TypeVar("F_contra", contravariant=True)


@dataclass(frozen=True)
class SolveInfo:
    """The outcome of a solve.

    Parameters
    ----------
    converged : bool
        Whether the solution is valid.
    iterations : int
        Iterations taken; 0 for a direct solve.
    residual_norm : float or None
        Final residual norm, when the solver computes one.
    message : str
        Why the solve failed, or ``""``.
    """

    converged: bool
    iterations: int = 0
    residual_norm: Optional[float] = None
    message: str = ""


@dataclass(frozen=True)
class SolverCapabilities:
    r"""What a solver can do, read before it is set up.

    Parameters
    ----------
    transpose : bool
        Whether its handles can solve with :math:`A^\top` (their
        ``transpose()`` is not ``None``).
    """

    transpose: bool


class SolveFailedError(RuntimeError):
    """Raised by a handle's ``apply`` when the solve does not converge."""


@runtime_checkable
class SolverHandleProtocol(ActionOperatorProtocol[V, MatrixFree], Protocol[V]):
    r"""A solver set up for an operator :math:`A`: an operator for
    :math:`A^{-1}`, from :math:`A`'s range to its domain."""

    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> tuple[V, SolveInfo]:
        """Solve :math:`A x = b` for a block of right-hand sides.

        Never raises on failure: check ``info.converged``. ``tol`` and
        ``x0`` are ignored by direct solvers.
        """
        ...

    def transpose(self) -> Optional["SolverHandleProtocol[V]"]:
        r"""A handle for :math:`A^\top`, or ``None`` if the solver has none."""
        ...


@runtime_checkable
class LinearSolverProtocol(Protocol, Generic[V, F_contra]):
    """A solver accepting operators whose format is ``F_contra``."""

    def capabilities(self) -> SolverCapabilities:
        """What handles of this solver can do."""
        ...

    def setup(self, op: ActionOperatorProtocol[V, F_contra]) -> SolverHandleProtocol[V]:
        """Prepare to solve with ``op`` (factor it, build a preconditioner).

        Raises
        ------
        TypeError
            If ``op``'s format is not the one this solver needs.
        ValueError
            If ``op`` is not square.
        """
        ...


def require_format(
    op: ActionOperatorProtocol[Any, Any], needed: type, solver: str
) -> Any:
    """Return ``op.format()``, raising unless it is a ``needed``.

    The runtime half of format pairing, for pairings decided by data that
    mypy cannot see.
    """
    fmt = op.format()
    if not isinstance(fmt, needed):
        raise TypeError(
            f"{solver} needs an operator in {needed.__name__}, got one in "
            f"{type(fmt).__name__}"
        )
    return fmt


def require_square(op: ActionOperatorProtocol[Any, Any], solver: str) -> None:
    if op.domain().size != op.range().size:
        raise ValueError(
            f"{solver} needs a square operator, got range size "
            f"{op.range().size} and domain size {op.domain().size}"
        )


class SolverHandleBase(ABC, Generic[V]):
    r"""The operator half of a handle, shared by every solver.

    A subclass supplies ``solve`` and ``transpose``. The handle's domain
    is the operator's range and its range the operator's domain; its
    properties are the operator's, since the inverse of a symmetric, SPD,
    diagonal or identity matrix is again one.

    Parameters
    ----------
    op : ActionOperatorProtocol
        The operator the handle inverts.
    """

    def __init__(self, op: ActionOperatorProtocol[V, Any]) -> None:
        self._op_domain = op.domain()
        self._op_range = op.range()
        self._properties = op.properties()

    def domain(self) -> SpaceDescriptor:
        return self._op_range

    def range(self) -> SpaceDescriptor:
        return self._op_domain

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> MatrixFree:
        return MatrixFree()

    @abstractmethod
    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> tuple[V, SolveInfo]: ...

    @abstractmethod
    def transpose(self) -> Optional["SolverHandleBase[V]"]: ...

    def apply(self, x: V, out: Optional[V] = None) -> V:
        """Return :math:`A^{-1} x`. ``out`` is ignored.

        Raises
        ------
        SolveFailedError
            If the solve does not converge.
        """
        solution, info = self.solve(x)
        if not info.converged:
            raise SolveFailedError(info.message)
        return solution

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        r"""Return :math:`A^{-\top} y`. ``out`` is ignored.

        Raises
        ------
        TypeError
            If the solver has no transpose.
        SolveFailedError
            If the solve does not converge.
        """
        transposed = self.transpose()
        if transposed is None:
            raise TypeError(f"{type(self).__name__} has no transposed solve")
        return transposed.apply(y)


class _DiagonalHandle(SolverHandleBase[V]):
    def __init__(
        self,
        op: ActionOperatorProtocol[V, Any],
        diagonal: V,
        space: VectorSpaceProtocol[V, Any],
    ) -> None:
        super().__init__(op)
        self._diagonal = diagonal
        self._space = space

    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> tuple[V, SolveInfo]:
        solution = self._space.divide_entries(rhs, self._diagonal)
        if not self._space.all_finite(solution):
            return solution, SolveInfo(False, message="zero diagonal entry")
        return solution, SolveInfo(True)

    def transpose(self) -> "_DiagonalHandle[V]":
        return self


class DiagonalSolver(Generic[V]):
    """Solves with an operator declared diagonal, in any format.

    Works for matrix-free operators: it needs only the diagonal, through
    :class:`DiagonalCapableProtocol`, and the space's pointwise division.

    Parameters
    ----------
    space : VectorSpaceProtocol
        The vector space of the operator's domain and range.
    """

    def __init__(self, space: VectorSpaceProtocol[V, Any]) -> None:
        self._space = space

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def setup(self, op: DiagonalCapableProtocol[V, Any]) -> SolverHandleProtocol[V]:
        """Raises ``TypeError`` unless ``op`` declares ``diagonal``."""
        if not op.properties().diagonal:
            raise TypeError("DiagonalSolver needs an operator declared diagonal")
        op.domain().require_same(op.range(), "range of a diagonal operator")
        self._space.descriptor().require_same(op.domain(), "space of DiagonalSolver")
        return _DiagonalHandle(op, op.diagonal(), self._space)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
