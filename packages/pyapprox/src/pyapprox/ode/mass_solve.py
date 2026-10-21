r"""Solves with an explicit stepper's mass, through a mass system.

An explicit step of :math:`M \dot y = f(y, t)` solves only with
:math:`M`, which does not change between steps. ``MassSolve`` sets up a
system's solver for its mass once and solves every right-hand side with
that handle.

The stepper-facing result is the matrix-era ``MassMatrixProtocol``,
because the steppers and the time-step sweeps still consume it: solves
and actions go through the system, and the matrix-forming and structural
queries (``as_matrix``, ``add_to``, ``is_diagonal``, ...) are answered by
the model's own mass.

``explicit_mass`` is what an explicit stepper calls at construction: it
refuses a singular mass and returns the stepper's source of the mass.
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
    r""":math:`M` of an explicit stepper, solved by a handle of a mass
    system's solver set up on first solve and reused thereafter.

    Inputs are 1-D vectors or 2-D blocks; a block goes to the operator
    and the solver whole.

    Parameters
    ----------
    residual : ODEResidualProtocol
        The stepper's residual; its ``mass_matrix()`` answers the
        matrix-forming and structural queries.
    system : MassSystemProtocol
        The mass operator and its solver.

    Raises
    ------
    TypeError
        If ``system`` does not satisfy ``MassSystemProtocol``.
    """

    def __init__(
        self,
        residual: ODEResidualProtocol[Array],
        system: MassSystemProtocol[Array, F],
    ) -> None:
        if not isinstance(system, MassSystemProtocol):
            raise TypeError(
                f"system must satisfy MassSystemProtocol, got {type(system).__name__}"
            )
        self._residual = residual
        self._operator: ActionOperatorProtocol[Array, F] = system.mass()
        self._solver: LinearSolverProtocol[Array, F] = system.solver()
        self._bkd: Backend[Array] = residual.bkd()
        self._handle: Optional[SolverHandleProtocol[Array]] = None

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
        return self._residual.mass_matrix().as_matrix()

    def add_to(self, matrix: Array) -> Array:
        return self._residual.mass_matrix().add_to(matrix)

    def is_diagonal(self) -> bool:
        return self._residual.mass_matrix().is_diagonal()

    def is_identity(self) -> bool:
        return self._residual.mass_matrix().is_identity()

    def is_singular(self) -> bool:
        return self._residual.mass_matrix().is_singular()

    def zero_rows(self) -> list[int]:
        return self._residual.mass_matrix().zero_rows()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(solver={self._solver!r})"


def explicit_mass(
    residual: ODEResidualProtocol[Array],
    system: Optional[MassSystemProtocol[Array, F]],
    stepper: str,
) -> Callable[[], MassMatrixProtocol[Array]]:
    """An explicit stepper's source of its mass.

    The residual's own ``mass_matrix`` when ``system`` is omitted, else a
    :class:`MassSolve` through ``system``.

    Raises
    ------
    TypeError
        If the residual's mass is singular: an explicit step solves with
        the mass, which a differential-algebraic model's mass does not
        allow. Such a model needs an implicit stepper.
    """
    mass = residual.mass_matrix()
    if mass.is_singular():
        raise TypeError(
            f"{stepper} solves with the mass, which is singular: rows "
            f"{mass.zero_rows()} are algebraic equations. Use an implicit "
            "stepper for a differential-algebraic model."
        )
    if system is None:
        return residual.mass_matrix
    return MassSolve(residual, system).mass_matrix
