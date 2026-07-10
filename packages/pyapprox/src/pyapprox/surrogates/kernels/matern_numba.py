"""Numba JIT-compiled scalar kernel evaluations for Matern family.

Each function evaluates k(x_i, x_j) for a single pair of points where
``x_i`` and ``x_j`` are ``(nvars,)`` arrays and ``params`` is a
``(nparams,)`` array of kernel parameters (exponentiated length scales).

These are designed to be passed as first-class functions into
numba-compiled algorithms (e.g. fused pivoted Cholesky).

A ``make_nugget_eval`` factory creates a numba-compatible wrapper that
adds ``nugget * delta(i,j)`` to any scalar kernel function.

Note: This module requires numba.  If numba is not available, importing
this module will raise ImportError, which callers handle gracefully.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.util.numba_compat import njit

_NumbaFn = Callable[
    [
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
    ],
    float,
]


@njit(cache=True)
def matern32_eval(
    xi: NDArray[np.floating[Any]],
    xj: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
) -> float:
    """Matern-3/2: k(x_i, x_j) = (1 + √3·r) exp(-√3·r)."""
    r_sq = 0.0
    for d in range(xi.shape[0]):
        diff = (xi[d] - xj[d]) / params[d]
        r_sq += diff * diff
    r = np.sqrt(3.0 * r_sq)
    return float((1.0 + r) * np.exp(-r))


@njit(cache=True)
def matern52_eval(
    xi: NDArray[np.floating[Any]],
    xj: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
) -> float:
    """Matern-5/2: k(x_i, x_j) = (1 + √5·r + 5/3·r²) exp(-√5·r)."""
    r_sq = 0.0
    for d in range(xi.shape[0]):
        diff = (xi[d] - xj[d]) / params[d]
        r_sq += diff * diff
    r = np.sqrt(5.0 * r_sq)
    return float((1.0 + r + r * r / 3.0) * np.exp(-r))


@njit(cache=True)
def sqexp_eval(
    xi: NDArray[np.floating[Any]],
    xj: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
) -> float:
    """Squared-exponential: k(x_i, x_j) = exp(-r²/2)."""
    r_sq = 0.0
    for d in range(xi.shape[0]):
        diff = (xi[d] - xj[d]) / params[d]
        r_sq += diff * diff
    return float(np.exp(-0.5 * r_sq))


@njit(cache=True)
def exponential_eval(
    xi: NDArray[np.floating[Any]],
    xj: NDArray[np.floating[Any]],
    params: NDArray[np.floating[Any]],
) -> float:
    """Exponential (Matern-1/2): k(x_i, x_j) = exp(-r)."""
    r_sq = 0.0
    for d in range(xi.shape[0]):
        diff = (xi[d] - xj[d]) / params[d]
        r_sq += diff * diff
    return float(np.exp(-np.sqrt(r_sq)))


_nugget_eval_cache: dict[int, _NumbaFn] = {}


def make_nugget_eval(inner_eval: _NumbaFn) -> _NumbaFn:
    """Create a nugget-wrapped scalar kernel evaluator.

    The returned function expects ``params[-1]`` to be the nugget value
    and ``params[:-1]`` to be the inner kernel's parameters.

    When ``xi`` and ``xj`` point to the same memory (diagonal evaluation),
    adds the nugget. Otherwise returns the inner kernel value unchanged.

    Results are cached by inner_eval identity to avoid repeated JIT compilation.
    """
    key = id(inner_eval)
    if key in _nugget_eval_cache:
        return _nugget_eval_cache[key]

    @njit(cache=True)
    def nugget_eval(
        xi: NDArray[np.floating[Any]],
        xj: NDArray[np.floating[Any]],
        params: NDArray[np.floating[Any]],
    ) -> float:
        inner_params = params[:-1]
        val = inner_eval(xi, xj, inner_params)
        r_sq = 0.0
        for d in range(xi.shape[0]):
            diff = xi[d] - xj[d]
            r_sq += diff * diff
        if r_sq == 0.0:
            val += params[-1]
        return float(val)

    result: _NumbaFn = nugget_eval
    _nugget_eval_cache[key] = result
    return result
