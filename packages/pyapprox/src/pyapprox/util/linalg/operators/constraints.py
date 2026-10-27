r"""Row-constrained operators: row replacement, row zeroing and symmetric
elimination.

With :math:`P` the projection that zeros the rows of a row set,

- :class:`RowReplacedOperator` is :math:`P A + (I - P)`: the rows in the
  set become identity rows and every other row of :math:`A` is kept,
  columns included. It is *not* the symmetric elimination
  :math:`P A P + (I - P)`, which also zeros the columns: the two give
  different solutions when the right-hand side is nonzero on the
  replaced rows, and their transposes differ on those rows;
- :class:`RowsZeroedOperator` is :math:`P A`: the rows in the set are
  zero. Its transpose, :math:`A^\top P`, zeros the columns instead;
- :class:`SymmetricEliminatedOperator` is :math:`P A P + (I - P)`: the
  rows and columns in the set zeroed and a unit diagonal on them, so it
  keeps :math:`A`'s symmetry and definiteness.

Neither is formed as a matrix to be applied: the action is :math:`A`'s
plus a fused row operation of the row set, costing one application of
:math:`A` and work proportional to the rows in the set. A stored
operator's edited storage is formed only when a consumer asks for the
format (a direct solver at setup), through the format's own row methods.

Row edits break symmetry, so the declared properties are cleared except
where the result is provably still structured (see
:func:`rows_replaced_properties`); symmetric elimination edits rows and
columns alike, so it keeps them (see
:func:`symmetric_eliminated_properties`).
"""

from typing import Generic, Optional, TypeVar

from pyapprox.util.linalg.operators.algebra import Transposed
from pyapprox.util.linalg.operators.descriptor import SpaceDescriptor
from pyapprox.util.linalg.operators.formats import FormatProtocol
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    OperatorProperties,
    RowSetProtocol,
)

V = TypeVar("V")
F = TypeVar("F", bound=FormatProtocol)


def rows_replaced_properties(
    props: OperatorProperties, nrows: int
) -> OperatorProperties:
    r"""Properties of :math:`P A + (I - P)` when the set has ``nrows`` rows.

    Unchanged for an empty set. Otherwise none, unless :math:`A` is
    diagonal: then the result is diagonal (its replaced entries are 1),
    hence symmetric, SPD if :math:`A` is, and the identity if :math:`A`
    is. A replaced row of a non-diagonal matrix makes it unsymmetric.
    """
    if nrows == 0:
        return props
    if not props.diagonal:
        return OperatorProperties()
    return OperatorProperties(
        symmetric=True, spd=props.spd, diagonal=True, identity=props.identity
    )


def rows_zeroed_properties(props: OperatorProperties, nrows: int) -> OperatorProperties:
    r"""Properties of :math:`P A` when the set has ``nrows`` rows.

    Unchanged for an empty set. Otherwise none, unless :math:`A` is
    diagonal: then the result is diagonal and symmetric, but singular, so
    neither SPD nor the identity.
    """
    if nrows == 0:
        return props
    if not props.diagonal:
        return OperatorProperties()
    return OperatorProperties(symmetric=True, diagonal=True)


def symmetric_eliminated_properties(
    props: OperatorProperties, nrows: int
) -> OperatorProperties:
    r"""Properties of :math:`P A P + (I - P)` when the set has ``nrows`` rows.

    Up to a permutation it is block diagonal: :math:`A`'s principal block
    on the rows outside the set and the identity on the set. A principal
    block of a symmetric, SPD, diagonal or identity matrix is one too, so
    each of those properties carries over; ``zero`` does not (the unit
    diagonal), unless the set is empty.
    """
    if nrows == 0:
        return props
    return OperatorProperties(
        symmetric=props.symmetric,
        spd=props.spd,
        diagonal=props.diagonal,
        identity=props.identity,
    )


def _require_rows_of(
    rows: RowSetProtocol[V], descriptor: SpaceDescriptor, what: str
) -> None:
    descriptor.require_same(rows.descriptor(), f"row set of {what}")


class RowReplacedOperator(Generic[V, F]):
    r"""The operator :math:`P A + (I - P)`.

    Parameters
    ----------
    op : ActionOperatorProtocol
        :math:`A`, mapping one space to itself.
    rows : RowSetProtocol
        The rows to replace, a row set of that space.

    Raises
    ------
    ValueError
        If ``op`` is not square in one space, or ``rows`` belongs to
        another space.
    """

    def __init__(
        self, op: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> None:
        op.domain().require_same(op.range(), "range of a row-replaced operator")
        _require_rows_of(rows, op.range(), "a row-replaced operator")
        self._op = op
        self._rows = rows
        self._properties = rows_replaced_properties(op.properties(), rows.count())
        self._format: Optional[F] = None

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The operand's storage with the rows replaced, computed on first
        request."""
        if self._format is None:
            self._format = self._op.format().rows_replaced(self._rows)
        return self._format

    def apply(self, x: V, out: Optional[V] = None) -> V:
        y = self._op.apply(x, out=out)
        return self._rows.replace_rows(y, x, out=y)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        r""":math:`A^\top P y + (I - P) y`."""
        z = self._op.apply_transpose(self._rows.zero_rows(y), out=out)
        return self._rows.add_rows(z, y, out=z)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._op!r}, nrows={self._rows.count()})"


class RowsZeroedOperator(Generic[V, F]):
    r"""The operator :math:`P A`.

    Parameters
    ----------
    op : ActionOperatorProtocol
        :math:`A`, of any shape.
    rows : RowSetProtocol
        The rows to zero, a row set of ``op``'s range.

    Raises
    ------
    ValueError
        If ``rows`` belongs to a space other than ``op``'s range.
    """

    def __init__(
        self, op: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> None:
        _require_rows_of(rows, op.range(), "a rows-zeroed operator")
        self._op = op
        self._rows = rows
        self._properties = rows_zeroed_properties(op.properties(), rows.count())
        self._format: Optional[F] = None

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The operand's storage with the rows zeroed, computed on first
        request."""
        if self._format is None:
            self._format = self._op.format().rows_zeroed(self._rows)
        return self._format

    def apply(self, x: V, out: Optional[V] = None) -> V:
        y = self._op.apply(x, out=out)
        return self._rows.zero_rows(y, out=y)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        r""":math:`A^\top P y`."""
        return self._op.apply_transpose(self._rows.zero_rows(y), out=out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._op!r}, nrows={self._rows.count()})"


class SymmetricEliminatedOperator(Generic[V, F]):
    r"""The operator :math:`P A P + (I - P)`.

    Its action and stored form are those of the row-replaced operator of
    :math:`A P` (:math:`A` with the set's columns zeroed), built from the
    row operators above and transposes, so the mechanics have one
    implementation: one application of :math:`A` plus row work, and a
    stored form computed only on request. What this class adds is the
    properties that composition cannot prove: symmetry and definiteness
    survive (see :func:`symmetric_eliminated_properties`), so a solver may
    use a symmetric or SPD factorization.

    Parameters
    ----------
    op : ActionOperatorProtocol
        :math:`A`, mapping one space to itself.
    rows : RowSetProtocol
        The rows and columns to eliminate, a row set of that space.

    Raises
    ------
    ValueError
        If ``op`` is not square in one space, or ``rows`` belongs to
        another space.
    """

    def __init__(
        self, op: ActionOperatorProtocol[V, F], rows: RowSetProtocol[V]
    ) -> None:
        op.domain().require_same(op.range(), "range of a symmetric elimination")
        _require_rows_of(rows, op.range(), "a symmetric elimination")
        self._op = op
        self._rows = rows
        # (P A^T)^T = A P: A with the set's columns zeroed.
        self._eliminated = RowReplacedOperator(
            Transposed(RowsZeroedOperator(Transposed(op), rows)), rows
        )
        self._properties = symmetric_eliminated_properties(
            op.properties(), rows.count()
        )

    def domain(self) -> SpaceDescriptor:
        return self._op.domain()

    def range(self) -> SpaceDescriptor:
        return self._op.range()

    def properties(self) -> OperatorProperties:
        return self._properties

    def format(self) -> F:
        """The operand's storage with the rows and columns eliminated,
        computed on first request."""
        return self._eliminated.format()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        return self._eliminated.apply(x, out=out)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        r""":math:`P A^\top P y + (I - P) y`."""
        return self._eliminated.apply_transpose(y, out=out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._op!r}, nrows={self._rows.count()})"
