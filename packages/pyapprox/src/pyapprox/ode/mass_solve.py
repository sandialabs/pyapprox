r"""Solves with an explicit stepper's mass, through a mass system.

An explicit step of :math:`M \dot y = f(y, t)` solves only with
:math:`M`, which does not change between steps. ``MassSolve`` sets up a
system's solver for its mass once and solves every right-hand side with
that handle.

The stepper-facing result is the matrix-era ``MassMatrixProtocol``,
because the steppers and the time-step sweeps still consume it.
Everything it answers comes from the system's mass operator, so a
stepper given a system never reads a second mass: solves and actions go
through the operator and its solver, and the matrix-forming and
structural queries (``as_matrix``, ``add_to``, ``zero_rows``, ...) come
from the matrix the operator's actions form, and from its declared
properties.

``explicit_mass`` is what an explicit stepper calls at construction: it
refuses a structurally singular mass and returns the stepper's source of
the mass.
"""

from typing import Callable, Generic, Optional, TypeVar

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.ode.protocols import MassSystemProtocol, ODEResidualProtocol
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import (
    ActionOperatorProtocol,
    LinearSolverProtocol,
    SolveFailedError,
    SolverHandleProtocol,
)

F = TypeVar("F")


class MassSolve(Generic[Array, F]):
    r""":math:`M` of a mass system, solved by a handle of the system's
    solver set up on first solve and reused thereafter.

    Inputs are 1-D vectors or 2-D blocks; a block goes to the operator
    and the solver whole. The matrix :math:`M` is formed on first request
    from the operator's action on the identity, and kept.

    Parameters
    ----------
    system : MassSystemProtocol
        The mass operator and its solver.
    bkd : Backend
        The backend of the operator's blocks.

    Raises
    ------
    TypeError
        If ``system`` does not satisfy ``MassSystemProtocol``.
    """

    def __init__(
        self, system: MassSystemProtocol[Array, F], bkd: Backend[Array]
    ) -> None:
        if not isinstance(system, MassSystemProtocol):
            raise TypeError(
                f"system must satisfy MassSystemProtocol, got {type(system).__name__}"
            )
        self._operator: ActionOperatorProtocol[Array, F] = system.mass()
        self._solver: LinearSolverProtocol[Array, F] = system.solver()
        self._bkd = bkd
        self._handle: Optional[SolverHandleProtocol[Array]] = None
        self._matrix: Optional[Array] = None

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """The mass as the stepper reads it: this object."""
        return self

    def _setup(self) -> SolverHandleProtocol[Array]:
        if self._handle is None:
            self._handle = self._solver.setup(self._operator)
        return self._handle

    def _block(self, action: Callable[[Array], Array], vec: Array) -> Array:
        """Apply a block ``action`` to a 1-D vector or a 2-D block."""
        bkd = self._bkd
        nrows = vec.shape[0]
        if vec.ndim == 1:
            return bkd.reshape(action(bkd.reshape(vec, (nrows, 1))), (nrows,))
        return action(vec)

    @staticmethod
    def _solved(handle: SolverHandleProtocol[Array]) -> Callable[[Array], Array]:
        """The handle's solve of a block, raising when it fails."""

        def solve(rhs: Array) -> Array:
            solution, info = handle.solve(rhs)
            if not info.converged:
                raise SolveFailedError(info.message)
            return solution

        return solve

    def solve(self, vec: Array) -> Array:
        """Solve :math:`M x = b`."""
        return self._block(self._solved(self._setup()), vec)

    def solve_transpose(self, vec: Array) -> Array:
        r"""Solve :math:`M^\top x = b`."""
        transposed = self._setup().transpose()
        if transposed is None:
            raise TypeError(f"{type(self._solver).__name__} has no transposed solve")
        return self._block(self._solved(transposed), vec)

    def apply(self, vec: Array) -> Array:
        return self._block(self._operator.apply, vec)

    def apply_transpose(self, vec: Array) -> Array:
        return self._block(self._operator.apply_transpose, vec)

    def as_matrix(self) -> Array:
        """:math:`M` as a dense backend array, formed once as
        :math:`M I` from the operator's action (one block apply)."""
        if self._matrix is None:
            self._matrix = self._operator.apply(
                self._bkd.eye(self._operator.domain().size)
            )
        return self._matrix

    def add_to(self, matrix: Array) -> Array:
        """:math:`M` + ``matrix``, for a dense ``matrix``."""
        return self.as_matrix() + matrix

    def is_diagonal(self) -> bool:
        return self._operator.properties().diagonal

    def is_identity(self) -> bool:
        return self._operator.properties().identity

    def is_singular(self) -> bool:
        """Whether :math:`M` has a zero row: structural, so False does not
        prove :math:`M` nonsingular."""
        return len(self.zero_rows()) > 0

    def zero_rows(self) -> list[int]:
        """The rows of :math:`M` that are entirely zero."""
        bkd = self._bkd
        row_sums = bkd.sum(bkd.abs(self.as_matrix()), axis=1)
        return [i for i in range(row_sums.shape[0]) if bkd.to_float(row_sums[i]) == 0.0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(solver={self._solver!r})"


def explicit_mass(
    residual: ODEResidualProtocol[Array],
    system: Optional[MassSystemProtocol[Array, F]],
    stepper: str,
) -> Callable[[], MassMatrixProtocol[Array]]:
    """An explicit stepper's source of its mass.

    The residual's own ``mass_matrix`` when ``system`` is omitted, else a
    :class:`MassSolve` of ``system``, whose mass then replaces the
    residual's everywhere.

    Raises
    ------
    ValueError
        If the mass the stepper will use has a zero row (structurally
        singular): an explicit step solves with the mass, which a
        differential-algebraic model's mass does not allow. Such a model
        needs an implicit stepper.
    """
    if system is None:
        source = residual.mass_matrix
    else:
        source = MassSolve(system, residual.bkd()).mass_matrix
    mass = source()
    if mass.is_singular():
        raise ValueError(
            f"{stepper} solves with the mass, which is singular: rows "
            f"{mass.zero_rows()} are algebraic equations. Use an implicit "
            "stepper for a differential-algebraic model."
        )
    return source
