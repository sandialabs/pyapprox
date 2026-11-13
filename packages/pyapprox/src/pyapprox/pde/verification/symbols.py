"""The sympy symbols verification expressions are written in.

Coordinates are ``x, y, z``, time is ``t``, the outward unit normal is
``nx, ny, nz``, and a nonlinear coefficient's state is ``u``. A stack
evaluates an expression by binding these symbols to its own points, times,
normals and states. A steady problem's expressions have no ``t``.
"""

from typing import Tuple

import sympy as sp

_COORDINATES = sp.symbols("x y z")
_NORMAL = sp.symbols("nx ny nz")
_STATE = sp.Symbol("u")
_TIME = sp.Symbol("t")


def _check_ndim(ndim: int) -> None:
    if not 1 <= ndim <= 3:
        raise ValueError(f"ndim must be 1, 2 or 3, got {ndim}")


def coordinates(ndim: int) -> Tuple[sp.Symbol, ...]:
    """The coordinate symbols of an ``ndim``-dimensional problem."""
    _check_ndim(ndim)
    return tuple(_COORDINATES[:ndim])


def normal(ndim: int) -> Tuple[sp.Symbol, ...]:
    """The outward unit normal symbols of an ``ndim``-dimensional
    problem."""
    _check_ndim(ndim)
    return tuple(_NORMAL[:ndim])


def state() -> sp.Symbol:
    """The state symbol of nonlinear coefficients, ``u``."""
    return _STATE


def time() -> sp.Symbol:
    """The time symbol, ``t``."""
    return _TIME
