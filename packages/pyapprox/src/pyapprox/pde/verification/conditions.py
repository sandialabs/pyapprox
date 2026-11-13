r"""Boundary conditions of verification problems, as mathematics.

A condition names a kind of boundary equation and builds its data as a
sympy expression from the problem's exact solution, so the data come from
the problem, not from the code under test. Nothing here evaluates; how a
discretization evaluates and enforces a condition is its own business.

With :math:`u^*` the exact solution, :math:`q(u^*)` the flux the problem
states (rows by component, columns by direction) and :math:`n` the
outward unit normal (symbols ``nx, ny, nz``):

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

from typing import Protocol, Tuple, TypeVar, Union, runtime_checkable

import sympy as sp

from pyapprox.pde.expressions.symbols import coordinates, time

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
    """The exact-solution expressions a condition's data are built from."""

    def solution(self) -> Tuple[sp.Expr, ...]:
        r""":math:`u^*`, one expression per component."""
        ...

    def normal_flux(self) -> Tuple[sp.Expr, ...]:
        r""":math:`q(u^*)\, n`, one expression per component, in the
        coordinate and normal symbols."""
        ...


@runtime_checkable
class ConditionProtocol(Protocol):
    """A kind of boundary equation, and its data."""

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
        """The right-hand side :math:`g`, one expression per component."""
        ...

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """The visitor's method for this kind, applied to this
        condition."""
        ...


class Dirichlet:
    r""":math:`u = u^*`."""

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
        r""":math:`u^*`."""
        return fields.solution()

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.dirichlet(self)``."""
        return visitor.dirichlet(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Neumann:
    r""":math:`q(u)\, n = q(u^*)\, n`."""

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
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
        expression = sp.sympify(alpha)
        allowed = set(coordinates(3)) | {time()}
        if not expression.free_symbols <= allowed:
            names = sorted(str(s) for s in expression.free_symbols - allowed)
            raise ValueError(f"alpha may depend only on x, y, z, t; got {names}")
        self._alpha = alpha
        self._expression: sp.Expr = expression

    def alpha(self) -> sp.Expr:
        r""":math:`\alpha`, in the coordinate and time symbols."""
        return self._expression

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
        r""":math:`q(u^*)\, n - \alpha u^*`."""
        return tuple(
            flux - self._expression * u
            for u, flux in zip(fields.solution(), fields.normal_flux())
        )

    def accept(self, visitor: ConditionVisitorProtocol[T]) -> T:
        """``visitor.robin(self)``."""
        return visitor.robin(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self._alpha!r})"
