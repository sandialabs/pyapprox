r"""Boundary conditions of verification problems, as mathematics.

A condition names a kind of boundary equation and builds its data as a
sympy expression from the problem's exact solution, so the data come from
the problem, not from the code under test. Nothing here evaluates; how a
discretization evaluates and enforces a condition is its own business.

A condition acts on one row (equation) of a problem. With :math:`u^*` the
exact solution of that row's field, :math:`q(u^*)` the flux of the row's
divergence terms and :math:`n` the outward unit normal (symbols ``n``
followed by each coordinate, e.g. ``nx, ny``):

- :class:`Dirichlet`: :math:`u = g`, :math:`g = u^*`;
- :class:`Neumann`: :math:`q(u)\, n = g`, :math:`g = q(u^*)\, n`;
- :class:`Robin`: :math:`q(u)\, n - \alpha u = g`, :math:`g = q(u^*)\, n
  - \alpha u^*`;
- :class:`Periodic`: the solution on this part equals that on the
  opposite part; no data. A problem must make both parts of a pair
  periodic, and the exact solution periodic across them; neither is
  checked here;
- :class:`Unprescribed`: no equation on this part; the row's own
  equation holds up to it. For a first-order (hyperbolic) row on a part
  its characteristics leave, such as an outflow. No data; whether the
  problem is then well posed is not checked here.

Natural data are written with the outward flux :math:`q\, n`. In Robin
the outward flux grows with :math:`u`, so a positive :math:`\alpha` is the
well-posed sign (for diffusion, :math:`q = -D\nabla u` and Robin reads
:math:`D\,\partial_n u + \alpha u = -g`), and ``Robin(0)`` is
``Neumann()``.

Time needs nothing here: the data depend on ``t`` exactly when the
solution or :math:`\alpha` does.

A stack enforces each kind its own way by implementing
:class:`ConditionVisitorProtocol`; ``condition.accept(visitor)`` calls the
visitor's method for that kind. A new kind adds a visitor method, so mypy
reports every stack that does not yet handle it.
"""

from typing import Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable

import sympy as sp

from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import parse

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ConditionVisitorProtocol(Protocol[T_co]):
    """What a stack builds for each kind of condition."""

    def dirichlet(self, condition: "Dirichlet") -> T_co:
        """The stack's object for a Dirichlet condition."""
        ...

    def neumann(self, condition: "Neumann") -> T_co:
        """The stack's object for a Neumann condition."""
        ...

    def robin(self, condition: "Robin") -> T_co:
        """The stack's object for a Robin condition."""
        ...

    def periodic(self, condition: "Periodic") -> T_co:
        """The stack's object for a periodic condition."""
        ...

    def unprescribed(self, condition: "Unprescribed") -> T_co:
        """The stack's object for a part that prescribes nothing."""
        ...


@runtime_checkable
class BoundaryFieldsProtocol(Protocol):
    """The exact-solution expressions of one row that a condition's data
    are built from."""

    def solution(self) -> sp.Expr:
        r""":math:`u^*` of the row's field."""
        ...

    def normal_flux(self) -> sp.Expr:
        r""":math:`q(u^*)\, n` of the row, in the coordinate, time and
        normal symbols."""
        ...

    def names(self) -> Names:
        """What the problem's expressions may use."""
        ...


@runtime_checkable
class ConditionProtocol(Protocol):
    """A kind of boundary equation, and its data."""

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """The coefficients the condition is written with, parsed (none
        for most kinds)."""
        ...

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        """The right-hand side :math:`g` on the row; ``None`` for a kind
        without data."""
        ...

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """The visitor's method for this kind, applied to this
        condition."""
        ...


class Dirichlet:
    r""":math:`u = u^*`."""

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """None."""
        return ()

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        r""":math:`u^*`."""
        return fields.solution()

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.dirichlet(self)``."""
        return visitor.dirichlet(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Neumann:
    r""":math:`q(u)\, n = q(u^*)\, n`."""

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """None."""
        return ()

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        r""":math:`q(u^*)\, n`."""
        return fields.normal_flux()

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.neumann(self)``."""
        return visitor.neumann(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Robin:
    r""":math:`q(u)\, n - \alpha u = q(u^*)\, n - \alpha u^*`; positive
    :math:`\alpha` is well posed.

    Parameters
    ----------
    alpha : float or str
        :math:`\alpha`: a number, or an expression in the problem's
        coordinates, ``t`` and constants (checked by the problem).
    """

    def __init__(self, alpha: Union[float, str]) -> None:
        self._alpha = alpha

    def alpha(self, names: Names) -> sp.Expr:
        r""":math:`\alpha`, parsed against ``names``."""
        return parse(self._alpha, names)

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        r""":math:`(\alpha,)`."""
        return (self.alpha(names),)

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        r""":math:`q(u^*)\, n - \alpha u^*`."""
        alpha = self.alpha(fields.names())
        result: sp.Expr = fields.normal_flux() - alpha * fields.solution()
        return result

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.robin(self)``."""
        return visitor.robin(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self._alpha!r})"


class Periodic:
    r"""The solution here equals that on the opposite part; no data."""

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """None."""
        return ()

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        """None: periodicity has no data."""
        return None

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.periodic(self)``."""
        return visitor.periodic(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Unprescribed:
    r"""No equation on this part: the row's own equation holds up to it.

    Stated explicitly, rather than by leaving the part out, so that a
    problem still names a condition on every part of every boundary row
    and a forgotten one is refused.
    """

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """None."""
        return ()

    def data(self, fields: BoundaryFieldsProtocol) -> Optional[sp.Expr]:
        """None: nothing is prescribed."""
        return None

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.unprescribed(self)``."""
        return visitor.unprescribed(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
