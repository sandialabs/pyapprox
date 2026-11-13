r"""The terms a verification problem's strong form is built from.

Each term acts on one row (equation), named after the field it governs,
and is written over the coordinates, ``t``, the jet symbols of the
problem's fields and its named constants
(:mod:`pyapprox.pde.expressions.symbols`). A row's strong form is the sum
of its terms,

.. math::

    \sum_k m_k\, \partial_t^k u + \nabla\cdot q + b = f,

and its forcing :math:`f` is that sum at the exact solution. A row
without a time term is algebraic in time (a DAE row); a problem without
space (an ODE) has only time terms and sources.

- :class:`TimeDerivative`: :math:`m\,\partial_t^k u_{\mathrm{row}}`,
  :math:`k = 1` or 2.
- :class:`Divergence`: :math:`\nabla\cdot q`; natural boundary conditions
  prescribe the :math:`q` of a row's divergence terms.
- :class:`Source`: :math:`b`, everything else.

Terms keep what they were given and are parsed against the problem's
names, so a constant the problem declares (``E``) means that constant.
"""

from typing import Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import sympy as sp

from pyapprox.pde.expressions.symbols import (
    check_field_name,
    coordinates,
    jets,
    time,
)
from pyapprox.util.expressions.names import Names
from pyapprox.util.expressions.parsing import Value, parse


class JetSubstitution:
    r"""Replaces jet symbols by the exact solution and its derivatives.

    Parameters
    ----------
    solution : Mapping[str, sympy.Expr]
        The exact solution, by field name, in the coordinates and ``t``.
    ndim : int
        The spatial dimension.
    """

    def __init__(self, solution: Mapping[str, sp.Expr], ndim: int) -> None:
        coords = dict(zip("xyz", coordinates(ndim)))
        self._replacements: Dict[sp.Symbol, sp.Expr] = {}
        for name, value in solution.items():
            for symbol, directions in jets(name, ndim).items():
                derived = value
                for direction in directions:
                    derived = sp.diff(derived, coords[direction])
                self._replacements[symbol] = derived
        self._ndim = ndim

    def ndim(self) -> int:
        """The spatial dimension."""
        return self._ndim

    def __call__(self, expression: sp.Expr) -> sp.Expr:
        """``expression`` with every jet symbol replaced."""
        result: sp.Expr = expression.xreplace(self._replacements)
        return result


@runtime_checkable
class TermProtocol(Protocol):
    """One term of one row of a strong form."""

    def row(self) -> str:
        """The row (field name) the term belongs to."""
        ...

    def time_order(self) -> int:
        """The order of its time derivative; 0 for none."""
        ...

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """Every expression the term is written with, parsed."""
        ...

    def flux(self, names: Names) -> Optional[Tuple[sp.Expr, ...]]:
        """:math:`q` per direction, parsed, for a divergence term;
        ``None`` for any other."""
        ...

    def residual(self, names: Names, jet: JetSubstitution) -> sp.Expr:
        """The term at the exact solution."""
        ...


class TimeDerivative:
    r""":math:`m\,\partial_t^k u_{\mathrm{row}}`.

    Parameters
    ----------
    row : str
        The field differentiated, and the row the term belongs to.
    order : int
        :math:`k`, 1 or 2.
    coefficient : expression, optional
        :math:`m`. Default 1.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2, or ``row`` cannot name a field.
    """

    def __init__(self, row: str, order: int = 1, coefficient: Value = 1) -> None:
        check_field_name(row)
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        self._row, self._order, self._coefficient = row, order, coefficient

    def row(self) -> str:
        """The row."""
        return self._row

    def time_order(self) -> int:
        """:math:`k`."""
        return self._order

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """:math:`(m,)`."""
        return (parse(self._coefficient, names),)

    def flux(self, names: Names) -> Optional[Tuple[sp.Expr, ...]]:
        """None: not a divergence term."""
        return None

    def residual(self, names: Names, jet: JetSubstitution) -> sp.Expr:
        r""":math:`m(u^*)\,\partial_t^k u^*_{\mathrm{row}}`."""
        (coefficient,) = self.expressions(names)
        value = jet(sp.Symbol(self._row))
        result: sp.Expr = jet(coefficient) * sp.diff(value, time(), self._order)
        return result

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._row!r}, order={self._order}, "
            f"coefficient={self._coefficient!r})"
        )


class Divergence:
    r""":math:`\nabla\cdot q`.

    Parameters
    ----------
    row : str
        The row the term belongs to.
    flux : Sequence
        :math:`q`, one expression per direction.

    Raises
    ------
    ValueError
        If ``row`` cannot name a field or ``flux`` is empty.
    """

    def __init__(self, row: str, flux: Sequence[Value]) -> None:
        check_field_name(row)
        if len(flux) == 0:
            raise ValueError("flux needs one entry per direction")
        self._row, self._flux = row, tuple(flux)

    def row(self) -> str:
        """The row."""
        return self._row

    def time_order(self) -> int:
        """0."""
        return 0

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """:math:`q`."""
        return tuple(parse(q, names) for q in self._flux)

    def flux(self, names: Names) -> Optional[Tuple[sp.Expr, ...]]:
        """:math:`q`, per direction."""
        return self.expressions(names)

    def residual(self, names: Names, jet: JetSubstitution) -> sp.Expr:
        r""":math:`\nabla\cdot q(u^*)`.

        Raises
        ------
        ValueError
            If ``q`` does not have one entry per direction.
        """
        ndim = jet.ndim()
        if len(self._flux) != ndim:
            raise ValueError(
                f"the flux of row {self._row!r} has {len(self._flux)} entries; "
                f"the problem has {ndim} dimensions"
            )
        return sp.Add(
            *(
                sp.diff(jet(q), x)
                for q, x in zip(self.expressions(names), coordinates(ndim))
            )
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._row!r}, {list(self._flux)!r})"


class Source:
    r""":math:`b`.

    Parameters
    ----------
    row : str
        The row the term belongs to.
    term : expression
        :math:`b`.

    Raises
    ------
    ValueError
        If ``row`` cannot name a field.
    """

    def __init__(self, row: str, term: Value) -> None:
        check_field_name(row)
        self._row, self._term = row, term

    def row(self) -> str:
        """The row."""
        return self._row

    def time_order(self) -> int:
        """0."""
        return 0

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """:math:`(b,)`."""
        return (parse(self._term, names),)

    def flux(self, names: Names) -> Optional[Tuple[sp.Expr, ...]]:
        """None: not a divergence term."""
        return None

    def residual(self, names: Names, jet: JetSubstitution) -> sp.Expr:
        """:math:`b(u^*)`."""
        return jet(self.expressions(names)[0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._row!r}, {self._term!r})"
