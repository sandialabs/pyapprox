"""The sympy symbols PDE expressions are written in.

Coordinates are single letters named by the domain: ``x, y, z`` for
physical space (:func:`standard`), others for phase space (``x, v``;
``x, y, z, u, v, w``). Time is ``t``; the outward unit normal is ``n``
followed by each coordinate (``nx, ny``; ``nx, nv``). Fields (solution
components) are named by the caller, and expressions use their jet
symbols: for a field ``f``, ``f`` itself, ``f_x``, ``f_v`` (first
derivatives), ``f_xv`` (mixed), and so on, directions in coordinate
order. A steady expression has no ``t``; a problem without space (an
ODE) has no coordinates.

:func:`pde_names` declares all of these, with any named constants and
extra functions, as :class:`~pyapprox.util.expressions.names.Names`, so
text is parsed by :func:`~pyapprox.util.expressions.parsing.parse` with
declared names winning over sympy's (a constant ``E`` is the caller's).
"""

import itertools
import re
from typing import Dict, Mapping, Optional, Sequence, Tuple

import sympy as sp

from pyapprox.util.expressions.names import CustomFunction, Names

_STANDARD = ("x", "y", "z")
_TIME = sp.Symbol("t")
# No underscores: in a jet symbol the underscore separates the field from
# its derivative directions.
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# Highest derivative order declared by default: fourth-order problems
# (beams, plates) write u_xxxx.
_JET_ORDER = 4


def standard(ndim: int) -> Tuple[str, ...]:
    """The coordinate names of ``ndim``-dimensional physical space,
    ``x, y, z`` (none for ``ndim = 0``)."""
    if not 0 <= ndim <= 3:
        raise ValueError(f"physical space has 0 to 3 dimensions, got {ndim}")
    return _STANDARD[:ndim]


def check_coordinates(coords: Sequence[str]) -> None:
    """Refuse coordinate names that are not distinct single letters other
    than ``t`` and ``n``.

    Raises
    ------
    ValueError
        If a name cannot be a coordinate, or one repeats.
    """
    for name in coords:
        if len(name) != 1 or not name.isalpha() or name in ("t", "n"):
            raise ValueError(
                f"coordinates are single letters other than t and n, got {name!r}"
            )
    if len(set(coords)) != len(coords):
        raise ValueError(f"coordinates must differ, got {list(coords)}")


def coordinates(coords: Sequence[str]) -> Tuple[sp.Symbol, ...]:
    """The coordinate symbols."""
    check_coordinates(coords)
    return tuple(sp.Symbol(name) for name in coords)


def normal(coords: Sequence[str]) -> Tuple[sp.Symbol, ...]:
    """The outward unit normal symbols, ``n`` followed by each
    coordinate."""
    check_coordinates(coords)
    return tuple(sp.Symbol(f"n{name}") for name in coords)


def time() -> sp.Symbol:
    """The time symbol, ``t``."""
    return _TIME


def check_field_name(name: str) -> None:
    """Refuse a field name that is not letters and digits starting with a
    letter, or that is ``t``. Clashes with a problem's coordinates and
    normals are refused by :func:`pde_names`.

    Raises
    ------
    ValueError
        If ``name`` cannot name a field.
    """
    if not _FIELD_NAME.fullmatch(name) or name == "t":
        raise ValueError(
            f"a field name is letters and digits, starting with a letter, and "
            f"not t; got {name!r}"
        )


def derivative(
    name: str, directions: str = "", coords: Sequence[str] = _STANDARD
) -> sp.Symbol:
    """The jet symbol of field ``name`` differentiated along
    ``directions`` (e.g. ``"xy"``; empty for the value itself), ordered as
    ``coords``.

    Raises
    ------
    ValueError
        If ``name`` cannot name a field or a direction is not a
        coordinate.
    """
    check_field_name(name)
    if any(d not in coords for d in directions):
        raise ValueError(f"directions are made of {list(coords)}; got {directions!r}")
    if not directions:
        return sp.Symbol(name)
    ordered = "".join(sorted(directions, key=list(coords).index))
    return sp.Symbol(f"{name}_{ordered}")


def gradient(name: str, coords: Sequence[str]) -> Tuple[sp.Symbol, ...]:
    """The first-derivative jet symbols of field ``name``."""
    return tuple(derivative(name, d, coords) for d in coords)


def jets(
    name: str, coords: Sequence[str], order: int = _JET_ORDER
) -> Dict[sp.Symbol, str]:
    """Every jet symbol of field ``name`` up to ``order``, with its
    directions (``""`` for the value)."""
    check_coordinates(coords)
    table = {derivative(name, "", coords): ""}
    for k in range(1, order + 1):
        for directions in itertools.combinations_with_replacement(coords, k):
            table[derivative(name, "".join(directions), coords)] = "".join(directions)
    return table


def pde_names(
    coords: Sequence[str],
    fields: Sequence[str] = (),
    constants: Optional[Mapping[str, float]] = None,
    functions: Optional[Mapping[str, CustomFunction]] = None,
    jet_order: int = _JET_ORDER,
) -> Names:
    """The names PDE expressions may use: coordinates, ``t``, normals, the
    jets of ``fields`` up to ``jet_order``, ``constants`` and
    ``functions``.

    Raises
    ------
    ValueError
        If a field cannot name one, or a field, constant or function
        clashes with a coordinate, ``t``, a normal or a field's jet.
    """
    geometry = list(coordinates(coords)) + [time()] + list(normal(coords))
    taken = {str(s) for s in geometry}
    for name in fields:
        check_field_name(name)
        if name in taken:
            raise ValueError(
                f"field {name!r} is a coordinate, t or a normal of {list(coords)}"
            )
    variables = list(geometry)
    for name in fields:
        variables += list(jets(name, coords, jet_order))
    taken = {str(s) for s in variables}
    for name in list(constants or {}) + list(functions or {}):
        if name in taken:
            raise ValueError(
                f"{name!r} is a coordinate, time, normal or field jet; "
                "constants and functions need other names"
            )
    return Names([str(s) for s in variables], constants, functions)
