r"""Quantities of interest a verification study can measure.

A quantity names a functional of one field's solution:

- :class:`PointValue`: :math:`u(x_0)`;
- :class:`Integral`: :math:`\int_\Omega u \,dx`;
- :class:`BoundaryFlux`: :math:`\int_\Gamma q(u)\, n \,ds` over one
  boundary part, with :math:`q` the flux of the row's divergence terms and
  :math:`n` the outward unit normal.

A quantity is data: it evaluates nothing. A stack computes it from its
own solution by implementing :class:`QuantityVisitorProtocol`;
``quantity.accept(visitor)`` calls the visitor's method for that kind. A
new kind adds a visitor method, so mypy reports every stack that does not
yet compute it. The exact value from a problem's exact solution is
:func:`pyapprox.pde.verification.exact.exact_value`.
"""

from typing import Protocol, Sequence, Tuple, TypeVar, runtime_checkable

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class QuantityVisitorProtocol(Protocol[T_co]):
    """What a stack computes for each kind of quantity."""

    def point_value(self, quantity: "PointValue") -> T_co:
        """The stack's value of a point value."""
        ...

    def integral(self, quantity: "Integral") -> T_co:
        """The stack's value of a domain integral."""
        ...

    def boundary_flux(self, quantity: "BoundaryFlux") -> T_co:
        """The stack's value of a boundary flux."""
        ...


@runtime_checkable
class QuantityProtocol(Protocol):
    """A functional of one field's solution."""

    def accept(self, visitor: QuantityVisitorProtocol[T]) -> T:
        """The visitor's method for this kind, applied to this quantity."""
        ...


class PointValue:
    r""":math:`u(x_0)` of one field.

    Parameters
    ----------
    field : str
        The field.
    point : Sequence[float]
        :math:`x_0`, one value per coordinate of the problem.

    Raises
    ------
    ValueError
        If ``point`` is empty.
    """

    def __init__(self, field: str, point: Sequence[float]) -> None:
        if len(point) == 0:
            raise ValueError("a point needs at least one coordinate")
        self._field = field
        self._point = tuple(float(value) for value in point)

    def field(self) -> str:
        """The field."""
        return self._field

    def point(self) -> Tuple[float, ...]:
        r""":math:`x_0`."""
        return self._point

    def accept(self, visitor: QuantityVisitorProtocol[T]) -> T:
        """``visitor.point_value(self)``."""
        return visitor.point_value(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._field!r}, {list(self._point)})"


class Integral:
    r""":math:`\int_\Omega u \,dx` of one field.

    Parameters
    ----------
    field : str
        The field.
    """

    def __init__(self, field: str) -> None:
        self._field = field

    def field(self) -> str:
        """The field."""
        return self._field

    def accept(self, visitor: QuantityVisitorProtocol[T]) -> T:
        """``visitor.integral(self)``."""
        return visitor.integral(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._field!r})"


class BoundaryFlux:
    r""":math:`\int_\Gamma q(u)\, n \,ds` of one row over one boundary part.

    Parameters
    ----------
    row : str
        The row, which must have divergence terms.
    part : str
        The boundary part :math:`\Gamma`.
    """

    def __init__(self, row: str, part: str) -> None:
        self._row, self._part = row, part

    def row(self) -> str:
        """The row."""
        return self._row

    def part(self) -> str:
        r"""The boundary part :math:`\Gamma`."""
        return self._part

    def accept(self, visitor: QuantityVisitorProtocol[T]) -> T:
        """``visitor.boundary_flux(self)``."""
        return visitor.boundary_flux(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._row!r}, {self._part!r})"
