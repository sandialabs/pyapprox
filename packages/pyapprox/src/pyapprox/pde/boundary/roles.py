"""Split a boundary-condition list into its two roles.

A BC is either a term added to ``F`` (``WeakFormBCProtocol``) or an
essential constraint on rows (``EssentialBCProtocol``). Splitting by
filtering on each role separately would silently drop anything that is
neither --- a multipoint or contact constraint, or a BC missing one
protocol method --- and the problem would be solved without it. This
split raises instead, naming the object.
"""

from typing import Generic, List, Sequence

from pyapprox.pde.boundary.protocols import (
    EssentialBCProtocol,
    WeakFormBCProtocol,
)
from pyapprox.util.backends.protocols import Array


class BCRoles(Generic[Array]):
    """A BC list split by role, each in the original list order.

    Parameters
    ----------
    terms : list of WeakFormBCProtocol
        Natural BCs, added to the spatial operator.
    essentials : list of EssentialBCProtocol
        Essential BCs, aggregated into a constraint set.
    """

    def __init__(
        self,
        terms: List[WeakFormBCProtocol[Array]],
        essentials: List[EssentialBCProtocol[Array]],
    ) -> None:
        self._terms = terms
        self._essentials = essentials

    def terms(self) -> List[WeakFormBCProtocol[Array]]:
        """Return the natural BCs."""
        return list(self._terms)

    def essentials(self) -> List[EssentialBCProtocol[Array]]:
        """Return the essential BCs."""
        return list(self._essentials)


def split_by_role(bcs: Sequence[object]) -> BCRoles[Array]:
    """Split ``bcs`` into natural terms and essential constraints.

    Parameters
    ----------
    bcs : sequence
        Boundary conditions, each satisfying exactly one of
        ``WeakFormBCProtocol`` and ``EssentialBCProtocol``.

    Raises
    ------
    TypeError
        If a BC satisfies neither role (it would otherwise be dropped)
        or both (its role would depend on the order of the checks).
    """
    terms: List[WeakFormBCProtocol[Array]] = []
    essentials: List[EssentialBCProtocol[Array]] = []
    for index, bc in enumerate(bcs):
        if isinstance(bc, WeakFormBCProtocol):
            if isinstance(bc, EssentialBCProtocol):
                raise TypeError(
                    f"boundary condition {index} ({bc!r}) satisfies both "
                    "WeakFormBCProtocol and EssentialBCProtocol; a BC has "
                    "exactly one role"
                )
            terms.append(bc)
        elif isinstance(bc, EssentialBCProtocol):
            essentials.append(bc)
        else:
            raise TypeError(
                f"boundary condition {index} ({bc!r}) satisfies neither "
                "WeakFormBCProtocol (a term added to the residual) nor "
                "EssentialBCProtocol (a constraint on rows), so it would "
                "be silently ignored. Implement one of them."
            )
    return BCRoles(terms, essentials)
