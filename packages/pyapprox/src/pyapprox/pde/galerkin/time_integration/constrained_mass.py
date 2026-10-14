"""The time integrator's mass, with the mass it was constrained from.

The Galerkin time-integration adapter steps with a mass whose essential
rows are replaced (identity rows) so that every stage system is the
constrained one. A quantity read on the essential rows themselves, such
as the time-integrated reaction at a Dirichlet boundary, needs those
rows as the scheme assembled them before the replacement: consistent or
lumped, whichever the scheme uses. :class:`ConstrainedMassMatrix` keeps
both, so such a quantity takes its mass from the scheme and can never
pair the scheme with a different mass.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.mass_matrix import MassMatrixProtocol
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class ConstrainedMassMatrixProtocol(MassMatrixProtocol[Array], Protocol):
    """A mass with essential rows replaced, and the mass before that."""

    def unconstrained(self) -> MassMatrixProtocol[Array]:
        """Return the mass before its essential rows were replaced:
        consistent or lumped, as the scheme assembled it."""
        ...


class ConstrainedMassMatrix(Generic[Array]):
    """The constrained mass, delegating to it, plus the unconstrained one.

    Parameters
    ----------
    constrained : MassMatrixProtocol
        The mass the time integrator steps with: identity at essential
        rows.
    unconstrained : MassMatrixProtocol
        The same mass before the essential rows were replaced.
    """

    def __init__(
        self,
        constrained: MassMatrixProtocol[Array],
        unconstrained: MassMatrixProtocol[Array],
    ) -> None:
        self._constrained = constrained
        self._unconstrained = unconstrained

    def unconstrained(self) -> MassMatrixProtocol[Array]:
        return self._unconstrained

    def apply(self, vec: Array) -> Array:
        return self._constrained.apply(vec)

    def solve(self, vec: Array) -> Array:
        return self._constrained.solve(vec)

    def apply_transpose(self, vec: Array) -> Array:
        return self._constrained.apply_transpose(vec)

    def solve_transpose(self, vec: Array) -> Array:
        return self._constrained.solve_transpose(vec)

    def as_matrix(self) -> Array:
        return self._constrained.as_matrix()

    def is_diagonal(self) -> bool:
        return self._constrained.is_diagonal()

    def is_identity(self) -> bool:
        return self._constrained.is_identity()

    def is_singular(self) -> bool:
        return self._constrained.is_singular()

    def zero_rows(self) -> list[int]:
        return self._constrained.zero_rows()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(constrained={self._constrained!r}, "
            f"unconstrained={self._unconstrained!r})"
        )
