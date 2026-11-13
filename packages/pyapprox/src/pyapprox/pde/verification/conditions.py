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
- :class:`Robin`: :math:`\alpha u + q(u)\, n = g`, :math:`g = \alpha u^*
  + q(u^*)\, n`.

Time needs nothing here: the data depend on ``t`` exactly when the
solution or :math:`\alpha` does.
"""

from typing import Protocol, Tuple, Union, runtime_checkable

import sympy as sp

from pyapprox.pde.verification.symbols import coordinates, time


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


class Dirichlet:
    r""":math:`u = u^*`."""

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
        r""":math:`u^*`."""
        return fields.solution()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Neumann:
    r""":math:`q(u)\, n = q(u^*)\, n`."""

    def data(self, fields: BoundaryFieldsProtocol) -> Tuple[sp.Expr, ...]:
        r""":math:`q(u^*)\, n`."""
        return fields.normal_flux()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Robin:
    r""":math:`\alpha u + q(u)\, n = \alpha u^* + q(u^*)\, n`.

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
        r""":math:`\alpha u^* + q(u^*)\, n`."""
        return tuple(
            self._expression * u + flux
            for u, flux in zip(fields.solution(), fields.normal_flux())
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(alpha={self._alpha!r})"
