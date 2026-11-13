"""The sympy symbols PDE expressions are written in.

Coordinates are ``x, y, z``, time is ``t``, the outward unit normal is
``nx, ny, nz``; these seven names are reserved. Fields (solution
components) are named by the caller, and expressions use their jet
symbols: for a field ``u``, ``u`` itself, ``u_x``, ``u_y`` (first
derivatives), ``u_xy`` (mixed), and so on, directions sorted ``x`` before
``y`` before ``z``. A steady expression has no ``t``; a problem without
space (an ODE) has ``ndim = 0`` and no coordinates.

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

_COORDINATES = sp.symbols("x y z")
_NORMAL = sp.symbols("nx ny nz")
_TIME = sp.Symbol("t")
_DIRECTIONS = "xyz"
_RESERVED = {"x", "y", "z", "t", "nx", "ny", "nz"}
# No underscores: in a jet symbol the underscore separates the field from
# its derivative directions.
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
# Highest derivative order declared by default: fourth-order problems
# (beams, plates) write u_xxxx.
_JET_ORDER = 4


def _check_ndim(ndim: int) -> None:
    if not 0 <= ndim <= 3:
        raise ValueError(f"ndim must be 0, 1, 2 or 3, got {ndim}")


def coordinates(ndim: int) -> Tuple[sp.Symbol, ...]:
    """The coordinate symbols of an ``ndim``-dimensional problem (none for
    ``ndim = 0``)."""
    _check_ndim(ndim)
    return tuple(_COORDINATES[:ndim])


def normal(ndim: int) -> Tuple[sp.Symbol, ...]:
    """The outward unit normal symbols of an ``ndim``-dimensional
    problem."""
    _check_ndim(ndim)
    return tuple(_NORMAL[:ndim])


def time() -> sp.Symbol:
    """The time symbol, ``t``."""
    return _TIME


def check_field_name(name: str) -> None:
    """Refuse a field name that is not letters and digits starting with a
    letter, or that is reserved.

    Raises
    ------
    ValueError
        If ``name`` cannot name a field.
    """
    if not _FIELD_NAME.fullmatch(name) or name in _RESERVED:
        raise ValueError(
            f"a field name is letters and digits, starting with a letter, and "
            f"not one of {sorted(_RESERVED)}; got {name!r}"
        )


def derivative(name: str, directions: str = "") -> sp.Symbol:
    """The jet symbol of field ``name`` differentiated along
    ``directions`` (e.g. ``"xy"``; empty for the value itself).

    Raises
    ------
    ValueError
        If ``name`` cannot name a field or ``directions`` has letters other
        than ``x``, ``y``, ``z``.
    """
    check_field_name(name)
    if any(d not in _DIRECTIONS for d in directions):
        raise ValueError(f"directions are made of x, y, z; got {directions!r}")
    if not directions:
        return sp.Symbol(name)
    ordered = "".join(sorted(directions, key=_DIRECTIONS.index))
    return sp.Symbol(f"{name}_{ordered}")


def gradient(name: str, ndim: int) -> Tuple[sp.Symbol, ...]:
    """The first-derivative jet symbols of field ``name``."""
    _check_ndim(ndim)
    return tuple(derivative(name, d) for d in _DIRECTIONS[:ndim])


def jets(
    name: str, ndim: int, order: int = _JET_ORDER
) -> Dict[sp.Symbol, str]:
    """Every jet symbol of field ``name`` up to ``order``, with its
    directions (``""`` for the value)."""
    _check_ndim(ndim)
    table = {derivative(name): ""}
    for k in range(1, order + 1):
        for directions in itertools.combinations_with_replacement(
            _DIRECTIONS[:ndim], k
        ):
            table[derivative(name, "".join(directions))] = "".join(directions)
    return table


def pde_names(
    ndim: int,
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
        If a field cannot name one, or a constant or function is reserved
        or clashes with a field's jet.
    """
    for name in fields:
        check_field_name(name)
    variables = list(coordinates(ndim)) + [time()] + list(normal(ndim))
    for name in fields:
        variables += list(jets(name, ndim, jet_order))
    taken = {str(s) for s in variables}
    for name in list(constants or {}) + list(functions or {}):
        if name in taken:
            raise ValueError(
                f"{name!r} is a coordinate, time, normal or field jet; "
                "constants and functions need other names"
            )
    return Names([str(s) for s in variables], constants, functions)
