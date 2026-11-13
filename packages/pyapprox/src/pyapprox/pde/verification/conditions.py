r"""Boundary conditions of verification problems, as mathematics.

A condition names a kind of boundary equation and builds its data as a
sympy expression from the problem's exact solution, so the data come from
the problem, not from the code under test. Nothing here evaluates; how a
discretization evaluates and enforces a condition is its own business.

A condition acts on one row (equation) of a problem. With :math:`u^*` the
exact solution of that row's field, :math:`q(u^*)` the flux of the row's
divergence terms and :math:`n` the outward unit normal (symbols ``nx,
ny, nz``):

- :class:`Dirichlet`: :math:`u = g`, :math:`g = u^*`;
- :class:`Neumann`: :math:`q(u)\, n = g`, :math:`g = q(u^*)\, n`;
- :class:`Robin`: :math:`q(u)\, n - \alpha u = g`, :math:`g = q(u^*)\, n
  - \alpha u^*`.

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

from typing import Protocol, TypeVar, Union, runtime_checkable

import sympy as sp

from pyapprox.pde.expressions.symbols import coordinates, pde_names, time
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


@runtime_checkable
class ConditionProtocol(Protocol):
    """A kind of boundary equation, and its data."""

    def data(self, fields: BoundaryFieldsProtocol) -> sp.Expr:
        """The right-hand side :math:`g` on the row."""
        ...

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """The visitor's method for this kind, applied to this
        condition."""
        ...


class Dirichlet:
    r""":math:`u = u^*`."""

    def data(self, fields: BoundaryFieldsProtocol) -> sp.Expr:
        r""":math:`u^*`."""
        return fields.solution()

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.dirichlet(self)``."""
        return visitor.dirichlet(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Neumann:
    r""":math:`q(u)\, n = q(u^*)\, n`."""

    def data(self, fields: BoundaryFieldsProtocol) -> sp.Expr:
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
        :math:`\alpha`: a constant, or an expression in ``x``, ``y``,
        ``z``, ``t``.

    Raises
    ------
    ValueError
        If ``alpha`` has symbols other than the coordinates and time.
    """

    def __init__(self, alpha: Union[float, str]) -> None:
        expression = parse(alpha, pde_names(3))
        allowed = set(coordinates(3)) | {time()}
        if not expression.free_symbols <= allowed:
            names = sorted(str(s) for s in expression.free_symbols - allowed)
            raise ValueError(f"alpha may depend only on x, y, z, t; got {names}")
        self._alpha = alpha
        self._expression: sp.Expr = expression

    def alpha(self) -> sp.Expr:
        r""":math:`\alpha`, in the coordinate and time symbols."""
        return self._expression

    def data(self, fields: BoundaryFieldsProtocol) -> sp.Expr:
        r""":math:`q(u^*)\, n - \alpha u^*`."""
        result: sp.Expr = fields.normal_flux() - self._expression * fields.solution()
        return result

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.robin(self)``."""
        return visitor.robin(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self._alpha!r})"
