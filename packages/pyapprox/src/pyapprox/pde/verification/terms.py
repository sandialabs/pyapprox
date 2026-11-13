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
- :class:`Source`: :math:`b`, everything local else.
- :class:`Moment`: a field integrated over some coordinates (a velocity
  moment of a kinetic distribution), the one nonlocal term.

Conservative and non-conservative forms are a choice of terms: the
conservative advection :math:`\nabla\cdot(a u)` is a divergence term whose
flux includes :math:`a u`; the non-conservative :math:`a\cdot\nabla u` is
a source.

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

    Derivatives are taken only for the jets an expression uses, so
    declaring high-order jets in many coordinates costs nothing.

    Parameters
    ----------
    solution : Mapping[str, sympy.Expr]
        The exact solution, by field name, in the coordinates and ``t``.
    coords : Sequence[str]
        The coordinate names.
    intervals : Mapping[str, Tuple[float, float]], optional
        ``(lower, upper)`` of each coordinate the domain is a product
        over, for terms that integrate.
    """

    def __init__(
        self,
        solution: Mapping[str, sp.Expr],
        coords: Sequence[str],
        intervals: Optional[Mapping[str, Tuple[float, float]]] = None,
    ) -> None:
        self._solution = dict(solution)
        self._coords = tuple(coords)
        self._symbols = dict(zip(coords, coordinates(coords)))
        self._intervals = dict(intervals or {})
        self._jets: Dict[sp.Symbol, Tuple[str, str]] = {}
        for name in self._solution:
            for symbol, directions in jets(name, coords).items():
                self._jets[symbol] = (name, directions)
        self._cache: Dict[sp.Symbol, sp.Expr] = {}

    def coordinates(self) -> Tuple[str, ...]:
        """The coordinate names."""
        return self._coords

    def ndim(self) -> int:
        """The number of coordinates."""
        return len(self._coords)

    def interval(self, coordinate: str) -> Tuple[float, float]:
        """``(lower, upper)`` of ``coordinate``.

        Raises
        ------
        ValueError
            If the domain is not a product over ``coordinate``.
        """
        if coordinate not in self._intervals:
            raise ValueError(
                f"the domain is not a product over {coordinate!r}, so nothing can "
                "be integrated over it"
            )
        return self._intervals[coordinate]

    def symbol(self, coordinate: str) -> sp.Symbol:
        """The symbol of ``coordinate``."""
        return self._symbols[coordinate]

    def _value(self, symbol: sp.Symbol) -> sp.Expr:
        if symbol not in self._cache:
            name, directions = self._jets[symbol]
            value = self._solution[name]
            for direction in directions:
                value = sp.diff(value, self._symbols[direction])
            self._cache[symbol] = value
        return self._cache[symbol]

    def __call__(self, expression: sp.Expr) -> sp.Expr:
        """``expression`` with every jet symbol replaced."""
        used = expression.free_symbols & set(self._jets)
        result: sp.Expr = expression.xreplace({s: self._value(s) for s in used})
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
        coords = jet.coordinates()
        if len(self._flux) != len(coords):
            raise ValueError(
                f"the flux of row {self._row!r} has {len(self._flux)} entries; "
                f"the problem has {len(coords)} coordinates"
            )
        return sp.Add(
            *(
                sp.diff(jet(q), jet.symbol(c))
                for q, c in zip(self.expressions(names), coords)
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


class Moment:
    r""":math:`\int w\, g \,d\xi`, a field integrated over some coordinates.

    Couples equations posed over different coordinates, e.g. the charge
    density :math:`\rho = \int f\, dv` that a Vlasov distribution
    :math:`f(x, v)` feeds to a Poisson equation in :math:`x`. Nonlocal: a
    stack evaluates it by quadrature over those coordinates.

    Parameters
    ----------
    row : str
        The row the term belongs to.
    integrand : expression
        :math:`w\, g`, e.g. ``"f"``, or ``"v*f"`` for a current.
    over : Sequence[str]
        The coordinates integrated over; the domain must be a product over
        each.

    Raises
    ------
    ValueError
        If ``row`` cannot name a field or ``over`` is empty.
    """

    def __init__(self, row: str, integrand: Value, over: Sequence[str]) -> None:
        check_field_name(row)
        if len(over) == 0:
            raise ValueError("a moment integrates over at least one coordinate")
        self._row, self._integrand, self._over = row, integrand, tuple(over)

    def row(self) -> str:
        """The row."""
        return self._row

    def over(self) -> Tuple[str, ...]:
        """The coordinates integrated over."""
        return self._over

    def time_order(self) -> int:
        """0."""
        return 0

    def expressions(self, names: Names) -> Tuple[sp.Expr, ...]:
        """The integrand."""
        return (parse(self._integrand, names),)

    def flux(self, names: Names) -> Optional[Tuple[sp.Expr, ...]]:
        """None: not a divergence term."""
        return None

    def residual(self, names: Names, jet: JetSubstitution) -> sp.Expr:
        """The integral at the exact solution, over the domain's
        intervals.

        Raises
        ------
        ValueError
            If the domain is not a product over a coordinate in ``over``.
        """
        value = jet(self.expressions(names)[0])
        limits = [(jet.symbol(c), *jet.interval(c)) for c in self._over]
        result: sp.Expr = sp.integrate(value, *limits)
        return result

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._row!r}, {self._integrand!r}, "
            f"over={list(self._over)})"
        )
