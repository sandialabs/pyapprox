"""
Backend-aware dispatch for smoothed AVaR computation.

Selects between two acceleration strategies based on backend type:
1. Numba parallel kernels (for NumPy backend when numba is installed)
2. Vectorized batch implementations (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so
that SampleAverageSmoothedAVaR is unaware of which strategy is active.
"""

from typing import Callable

import numpy as np

from pyapprox.risk.avar_compute import avar_jacobian_batch, avar_values_batch
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

HAS_NUMBA = package_available("numba")
if HAS_NUMBA:
    from pyapprox.risk.avar_numba import (
        avar_jacobian_numba,
        avar_values_numba,
    )

ValuesImpl = Callable[
    [Array, Array, Array, float, float, Backend[Array]], Array
]

JacobianImpl = Callable[
    [Array, Array, Array, Array, float, float, Backend[Array]], Array
]


def _is_numpy(bkd: Backend[Array]) -> bool:
    """Check if the backend is NumPy."""
    return isinstance(bkd, NumpyBkd)


def _wrap_numba_values(
    values: Array,
    weights: Array,
    alpha: Array,
    delta: float,
    lam: float,
    bkd: Backend[Array],
) -> Array:
    """Wrap numba kernel to match the batch signature."""
    weights_1d = np.asarray(weights).ravel()
    alpha_f = float(np.asarray(alpha).ravel()[0])
    result = avar_values_numba(
        np.asarray(values), weights_1d, alpha_f, delta, lam
    )
    return bkd.asarray(result[:, np.newaxis])


def _wrap_numba_jacobian(
    values: Array,
    jac_values: Array,
    weights: Array,
    alpha: Array,
    delta: float,
    lam: float,
    bkd: Backend[Array],
) -> Array:
    """Wrap numba kernel to match the batch signature."""
    weights_1d = np.asarray(weights).ravel()
    alpha_f = float(np.asarray(alpha).ravel()[0])
    result = avar_jacobian_numba(
        np.asarray(values),
        np.asarray(jac_values),
        weights_1d,
        alpha_f,
        delta,
        lam,
    )
    return bkd.asarray(result)


def get_avar_values_impl(
    bkd: Backend[Array],
) -> ValuesImpl[Array]:
    """Get the AVaR values implementation for the given backend.

    Dispatch order:
    1. NumPy + Numba installed -> Numba parallel kernel
    2. Otherwise -> vectorized batch fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (values, weights, alpha, delta, lam, bkd) -> Array
    """
    if _is_numpy(bkd) and HAS_NUMBA:
        return _wrap_numba_values
    return avar_values_batch


def get_avar_jacobian_impl(
    bkd: Backend[Array],
) -> JacobianImpl[Array]:
    """Get the AVaR Jacobian implementation for the given backend.

    Dispatch order:
    1. NumPy + Numba installed -> Numba parallel kernel
    2. Otherwise -> vectorized batch fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (values, jac_values, weights, alpha, delta, lam, bkd) -> Array
    """
    if _is_numpy(bkd) and HAS_NUMBA:
        return _wrap_numba_jacobian
    return avar_jacobian_batch
