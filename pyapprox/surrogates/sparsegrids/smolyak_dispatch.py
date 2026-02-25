"""Dispatch for Smolyak coefficient computation.

Selects between:
1. Numba fused kernel (if numba available) — binary search membership
2. Vectorized numpy with void dtype np.isin (fallback)

Both operate on raw numpy arrays. No torch dispatch needed since the
computation always converts to numpy (integer set membership, not
differentiable).
"""

from typing import Callable

import numpy as np

from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")


# Type alias: (np_indices, np_shifts, np_signs, nvars, nsubspaces, nshifts) -> np_coefs
SmolyakImpl = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, int, int], np.ndarray
]


def _generic_smolyak(
    np_indices: np.ndarray,
    np_shifts: np.ndarray,
    np_signs: np.ndarray,
    nvars: int,
    nsubspaces: int,
    nshifts: int,
) -> np.ndarray:
    """Compute Smolyak coefficients using vectorized np.isin with void dtype.

    Eliminates the inner Python loop by viewing rows as opaque byte strings
    and using np.isin for batch membership checking.
    """
    np_indices_T = np.ascontiguousarray(np_indices.T)  # (nsubspaces, nvars)
    row_size = nvars * np_indices_T.dtype.itemsize
    void_dt = np.dtype(f"V{row_size}")
    index_void = np_indices_T.view(void_dt).ravel()  # (nsubspaces,)

    np_coefs = np.zeros(nsubspaces, dtype=np.float64)

    for s in range(nshifts):
        shifted = np_indices + np_shifts[:, s : s + 1]  # (nvars, nsubspaces)
        shifted_T = np.ascontiguousarray(shifted.T)
        shifted_void = shifted_T.view(void_dt).ravel()
        mask = np.isin(shifted_void, index_void)
        np_coefs[mask] += np_signs[s]

    return np_coefs


def _make_numba_smolyak() -> SmolyakImpl:
    """Create a numba-backed Smolyak implementation."""
    from pyapprox.surrogates.sparsegrids.smolyak_numba import (
        smolyak_coefficients_numba,
    )

    def impl(
        np_indices: np.ndarray,
        np_shifts: np.ndarray,
        np_signs: np.ndarray,
        nvars: int,
        nsubspaces: int,
        nshifts: int,
    ) -> np.ndarray:
        return smolyak_coefficients_numba(
            np_indices, np_shifts, np_signs, nvars, nsubspaces, nshifts,
        )

    return impl


def get_smolyak_impl() -> SmolyakImpl:
    """Get the best available Smolyak coefficient implementation.

    Returns numba kernel if available, otherwise vectorized numpy fallback.
    """
    if _HAS_NUMBA:
        return _make_numba_smolyak()
    return _generic_smolyak
