r"""Manufactured verification problems, as sympy expressions.

A problem is an exact solution (one expression per field), a strong form
written as terms on rows (:mod:`pyapprox.pde.verification.terms`), a
domain, and conditions on its named boundary parts. It gives each row's
forcing (the strong form at the exact solution), flux and boundary data,
all as sympy expressions in the symbols of
:mod:`pyapprox.pde.expressions.symbols`. It evaluates nothing: a stack
binds the symbols to its own points, so the same problem verifies any
discretization, serial or distributed.

Named constants stay symbols in the terms and are replaced by their
values in everything the problem returns. A problem without a domain
(``ndim = 0``) is an ODE or DAE: no coordinates, no boundary. The
coordinates are the domain's, so kinetic equations pose on a phase space
(``x, v``) the same way PDEs pose on physical space.
"""

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import sympy as sp

from pyapprox.pde.expressions.symbols import (
    coordinates,
    jets,
    normal,
    pde_names,
    time,
)
from pyapprox.pde.verification.conditions import ConditionProtocol
from pyapprox.pde.verification.domains import DomainProtocol
from pyapprox.pde.verification.terms import JetSubstitution, TermProtocol
from pyapprox.util.expressions.names import CustomFunction, Names
from pyapprox.util.expressions.parsing import Value, parse

_Conditions = Union[ConditionProtocol, Mapping[str, ConditionProtocol]]


class _RowFields:
    """One row's exact solution and normal flux, for a condition; the flux
    is formed only if the condition asks (a row without divergence terms
    has none)."""

    def __init__(self, problem: "ManufacturedProblem", row: str) -> None:
        self._problem, self._row = problem, row

    def solution(self) -> sp.Expr:
        return self._problem.solution(self._row)

    def normal_flux(self) -> sp.Expr:
        return self._problem.normal_flux(self._row)

    def names(self) -> Names:
        return self._problem.names()


class ManufacturedProblem:
    r"""An exact solution, a strong form, a domain and conditions.

    Parameters
    ----------
    solution : Mapping[str, expression]
        :math:`u^*` by field name, in the coordinates, ``t`` and constants.
        The fields name the rows, in this order.
    terms : Sequence[TermProtocol]
        The strong form; every row needs at least one term.
    domain : DomainProtocol, optional
        Where the problem is posed; ``None`` for an ODE or DAE.
    conditions : Mapping, optional
        For every boundary part of ``domain``, by name: one condition for
        every boundary row, or a mapping from each boundary row to its
        condition. The boundary rows are those with divergence terms.
    constants : Mapping[str, float], optional
        Named constants and their values.
    functions : Mapping[str, CustomFunction], optional
        Extra functions the expressions call.

    Raises
    ------
    TypeError
        If ``domain``, a term or a condition is of the wrong type.
    ValueError
        If a term's row is not a field, a row has no term, the solution
        depends on anything but coordinates, ``t`` and constants, a flux
        has the wrong length, or the conditions do not cover exactly the
        boundary parts and boundary rows.
    """

    def __init__(
        self,
        solution: Mapping[str, Value],
        terms: Sequence[TermProtocol],
        domain: Optional[DomainProtocol] = None,
        conditions: Optional[Mapping[str, _Conditions]] = None,
        constants: Optional[Mapping[str, float]] = None,
        functions: Optional[Mapping[str, CustomFunction]] = None,
    ) -> None:
        if domain is not None and not isinstance(domain, DomainProtocol):
            raise TypeError(
                f"domain must satisfy DomainProtocol, got {type(domain).__name__}"
            )
        coords = () if domain is None else domain.coordinates()
        self._domain, self._coords = domain, coords
        self._fields = tuple(solution)
        self._names = pde_names(coords, self._fields, constants, functions)
        self._solution = {
            name: parse(value, self._names) for name, value in solution.items()
        }
        allowed = set(coordinates(coords)) | {time()} | set(self._names.constants())
        for name, expression in self._solution.items():
            extra = expression.free_symbols - allowed
            if extra:
                raise ValueError(
                    f"the solution of {name!r} may depend only on the "
                    f"coordinates, t and constants; got {sorted(map(str, extra))}"
                )
        self._terms = self._check_terms(terms)
        intervals: Dict[str, Tuple[float, float]] = {}
        for coordinate in coords:
            interval = None if domain is None else domain.interval(coordinate)
            if interval is not None:
                intervals[coordinate] = interval
        jet = JetSubstitution(self._solution, coords, intervals)
        self._forcing: Dict[str, sp.Expr] = {}
        self._flux: Dict[str, Tuple[sp.Expr, ...]] = {}
        self._boundary_rows: List[str] = []
        for row in self._fields:
            row_terms = [term for term in self._terms if term.row() == row]
            self._forcing[row] = sp.Add(
                *(term.residual(self._names, jet) for term in row_terms)
            )
            fluxes = [term.flux(self._names) for term in row_terms]
            present = [q for q in fluxes if q is not None]
            if present:
                self._flux[row] = tuple(
                    jet(sp.Add(*column)) for column in zip(*present)
                )
            own = set(jets(row, coords)) - {sp.Symbol(row)}
            used: "set[sp.Basic]" = set()
            for term in row_terms:
                for expression in term.expressions(self._names):
                    used |= expression.free_symbols
            if present or used & own:
                self._boundary_rows.append(row)
        self._conditions = self._check_conditions(conditions or {})

    def _check_terms(self, terms: Sequence[TermProtocol]) -> Tuple[TermProtocol, ...]:
        for term in terms:
            if not isinstance(term, TermProtocol):
                raise TypeError(
                    f"terms must satisfy TermProtocol, got {type(term).__name__}"
                )
            if term.row() not in self._fields:
                raise ValueError(
                    f"{term!r} acts on row {term.row()!r}; the fields are "
                    f"{list(self._fields)}"
                )
        empty = [row for row in self._fields if all(t.row() != row for t in terms)]
        if empty:
            raise ValueError(f"rows without terms: {empty}")
        return tuple(terms)

    def _check_conditions(
        self, conditions: Mapping[str, _Conditions]
    ) -> Mapping[str, Mapping[str, ConditionProtocol]]:
        parts = set() if self._domain is None else set(self._domain.boundaries())
        missing, unknown = parts - set(conditions), set(conditions) - parts
        if missing or unknown:
            raise ValueError(
                "conditions must name every boundary part of the domain and no "
                f"other: missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        rows = self.boundary_rows()
        allowed = (
            set(coordinates(self._coords)) | {time()} | set(self._names.constants())
        )
        checked: Dict[str, Mapping[str, ConditionProtocol]] = {}
        for part, given in conditions.items():
            if isinstance(given, Mapping):
                per_row = dict(given)
            else:
                per_row = {row: given for row in rows}
            if set(per_row) != set(rows):
                raise ValueError(
                    f"boundary {part!r} needs a condition on each boundary row "
                    f"{list(rows)}, got {sorted(per_row)}"
                )
            for row, condition in per_row.items():
                if not isinstance(condition, ConditionProtocol):
                    raise TypeError(
                        f"condition on {part!r}, row {row!r} must satisfy "
                        f"ConditionProtocol, got {type(condition).__name__}"
                    )
                for expression in condition.expressions(self._names):
                    extra = expression.free_symbols - allowed
                    if extra:
                        raise ValueError(
                            f"{condition!r} on {part!r} may depend only on the "
                            f"coordinates, t and constants; got "
                            f"{sorted(map(str, extra))}"
                        )
            checked[part] = MappingProxyType(per_row)
        return MappingProxyType(checked)

    def _values(self, expression: sp.Expr) -> sp.Expr:
        return self._names.substitute_constants(expression)

    def domain(self) -> Optional[DomainProtocol]:
        """Where the problem is posed; ``None`` for an ODE or DAE."""
        return self._domain

    def ndim(self) -> int:
        """The number of coordinates (0 for an ODE or DAE)."""
        return len(self._coords)

    def coordinates(self) -> Tuple[str, ...]:
        """The coordinate names."""
        return self._coords

    def fields(self) -> Tuple[str, ...]:
        """The field names, which name the rows."""
        return self._fields

    def names(self) -> Names:
        """What the problem's expressions may use."""
        return self._names

    def terms(self, row: str) -> Tuple[TermProtocol, ...]:
        """The terms of ``row``."""
        return tuple(term for term in self._terms if term.row() == row)

    def time_order(self, row: str) -> int:
        """The highest time derivative in ``row``; 0 for an algebraic
        row."""
        return max(term.time_order() for term in self.terms(row))

    def boundary_rows(self) -> Tuple[str, ...]:
        """The rows that take boundary conditions: those with a divergence
        term or a spatial derivative of their own field (a transport row
        in advective form). A row differentiating only other fields (a
        Stokes continuity row) takes none."""
        return tuple(self._boundary_rows)

    def is_transient(self) -> bool:
        """Whether a row has a time derivative or an expression depends on
        ``t``."""
        if any(term.time_order() > 0 for term in self._terms):
            return True
        expressions = list(self._solution.values()) + list(self._forcing.values())
        for part in self._conditions:
            expressions += list(self.boundary_data(part).values())
        return any(time() in e.free_symbols for e in expressions)

    def solution(self, row: str) -> sp.Expr:
        r""":math:`u^*` of field ``row``."""
        return self._values(self._solution[self._check_row(row)])

    def forcing(self, row: str) -> sp.Expr:
        """The forcing of ``row``: its strong form at the exact
        solution."""
        return self._values(self._forcing[self._check_row(row)])

    def flux(self, row: str) -> Optional[Tuple[sp.Expr, ...]]:
        r""":math:`q(u^*)` of ``row`` per direction; ``None`` without
        divergence terms."""
        flux = self._flux.get(self._check_row(row))
        return None if flux is None else tuple(self._values(q) for q in flux)

    def normal_flux(self, row: str) -> sp.Expr:
        r""":math:`q(u^*)\, n` of ``row``, in the coordinate, time and
        normal symbols.

        Raises
        ------
        ValueError
            If ``row`` has no divergence terms.
        """
        flux = self.flux(row)
        if flux is None:
            raise ValueError(f"row {row!r} has no divergence terms")
        return sp.Add(*(q * n for q, n in zip(flux, normal(self._coords))))

    def conditions(self, part: str) -> Mapping[str, ConditionProtocol]:
        """The condition on each boundary row of boundary part ``part``."""
        if part not in self._conditions:
            names = [] if self._domain is None else list(self._domain.boundaries())
            raise ValueError(f"no boundary part {part!r}; the domain has {names}")
        return self._conditions[part]

    def boundary_data(self, part: str) -> Dict[str, sp.Expr]:
        """The data of the conditions on boundary part ``part``, by row;
        rows whose condition has no data (periodic, unprescribed) are left
        out."""
        data: Dict[str, sp.Expr] = {}
        for row, condition in self.conditions(part).items():
            value = condition.data(_RowFields(self, row))
            if value is not None:
                data[row] = self._values(value)
        return data

    def _check_row(self, row: str) -> str:
        if row not in self._fields:
            raise ValueError(f"no row {row!r}; the fields are {list(self._fields)}")
        return row

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(fields={list(self._fields)}, "
            f"domain={self._domain!r})"
        )
