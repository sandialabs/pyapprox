r"""Format witnesses that need no linear-algebra library.

An operator's :meth:`format` returns a *format witness*: a value whose
type says how the operator is stored, and which, for a stored format,
gives access to the storage. A direct solver declares the format it
needs in its signature, so pairing it with an operator of another
format is a static type error. Stored formats belong to the stack that
owns their storage; only the storage-free format lives here.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Self, Sequence, runtime_checkable

from pyapprox.util.linalg.operators.protocols import RowSetProtocol


@runtime_checkable
class FormatProtocol(Protocol):
    """The storage algebra every format owns.

    Operator combinators (sums, transposes) build their format from their
    operands' formats through these methods, so each format decides how
    its storage combines and no combinator branches on the format.
    Results are computed only when a consumer asks for the format, so an
    operator that is only ever applied never assembles a combination.
    """

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, Self]]
    ) -> Self:
        """Return ``coefficient * self + sum(c * other for c, other in others)``."""
        ...

    def transposed(self) -> Self:
        """Return the format of the transposed operator."""
        ...

    def rows_replaced(self, rows: RowSetProtocol[Any]) -> Self:
        r"""Return the format of :math:`P A + (I - P)`: the rows in
        ``rows`` replaced by identity rows.

        A stored format accepts only its own stack's row sets and raises
        ``TypeError`` for another.
        """
        ...

    def rows_zeroed(self, rows: RowSetProtocol[Any]) -> Self:
        r"""Return the format of :math:`P A`: the rows in ``rows`` zeroed."""
        ...


@dataclass(frozen=True)
class MatrixFree:
    """The format of an operator known only through its action.

    It has no storage, so nothing that needs a matrix can be paired with
    an operator of this format. Its algebra is trivial: every combination
    of matrix-free operators is matrix-free.
    """

    def scaled_sum(
        self, coefficient: float, others: Sequence[tuple[float, "MatrixFree"]]
    ) -> "MatrixFree":
        return self

    def transposed(self) -> "MatrixFree":
        """The transpose of an unstored operator is unstored.

        This describes storage only: the transposed *action* comes from
        the operator (``Transposed.apply`` calls ``apply_transpose``).
        """
        return self

    def rows_replaced(self, rows: RowSetProtocol[Any]) -> "MatrixFree":
        return self

    def rows_zeroed(self, rows: RowSetProtocol[Any]) -> "MatrixFree":
        return self
