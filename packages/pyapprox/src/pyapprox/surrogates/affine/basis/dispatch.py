"""
Backend-aware dispatch for MultiIndexBasis tensor product assembly.

Selects between three acceleration strategies based on the backend type:
1. Numba fused kernels (for NumPy backend) — avoids all intermediate arrays
2. torch-native functions (for PyTorch backend)
3. Vectorized backend-agnostic implementations (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so that
MultiIndexBasis is unaware of which strategy is active.

All dispatched implementations are module-level functions (not closures) so
that objects storing them as attributes remain picklable.
"""

from functools import lru_cache
from typing import Callable, List, cast

import numpy as np

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")

from pyapprox.surrogates.affine.basis.compute import (
    basis_eval_vectorized,
    basis_hessian_vectorized,
    basis_jacobian_vectorized,
)

# Type aliases for dispatch callables
BasisEvalImpl = Callable[[List[Array], Array, int, Backend[Array]], Array]

BasisJacobianImpl = Callable[
    [List[Array], List[Array], Array, int, Backend[Array]], Array
]

BasisHessianImpl = Callable[
    [List[Array], List[Array], List[Array], Array, int, Backend[Array]], Array
]


def _is_numpy(bkd: Backend[Array]) -> bool:
    """Check if backend is NumPy."""
    return isinstance(bkd, NumpyBkd)


def _is_torch(bkd: Backend[Array]) -> bool:
    """Check if backend is PyTorch; import deferred to avoid torch load time."""
    from pyapprox.util.backends.torch import TorchBkd

    return isinstance(bkd, TorchBkd)


def _stack_1d_arrays_for_numba(
    vals_1d: List[np.ndarray],
) -> np.ndarray:
    """Stack list of variable-width 1D arrays into padded 3D array.

    Parameters
    ----------
    vals_1d : List[np.ndarray]
        Each element has shape (nsamples, nterms_1d_d).

    Returns
    -------
    np.ndarray
        Shape: (nvars, nsamples, max_nterms_1d). Zero-padded.
    """
    nvars = len(vals_1d)
    nsamples = vals_1d[0].shape[0]
    max_nterms = max(v.shape[1] for v in vals_1d)
    stacked = np.zeros((nvars, nsamples, max_nterms))
    for dd in range(nvars):
        stacked[dd, :, : vals_1d[dd].shape[1]] = vals_1d[dd]
    return stacked


# --- Numba implementations ---


def _numba_basis_eval(
    vals_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Numba-backed basis evaluation."""
    from pyapprox.surrogates.affine.basis.compute_numba import (
        basis_eval_numba,
    )

    stacked = _stack_1d_arrays_for_numba(vals_1d)
    indices_np = np.asarray(indices)
    nsamples = vals_1d[0].shape[0]
    nterms = indices_np.shape[1]
    result: Array = bkd.asarray(
        basis_eval_numba(
            stacked,
            indices_np,
            nvars,
            nsamples,
            nterms,
        )
    )
    return result


def _numba_basis_jacobian(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Numba-backed basis Jacobian."""
    from pyapprox.surrogates.affine.basis.compute_numba import (
        basis_jacobian_numba,
    )

    stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
    stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
    indices_np = np.asarray(indices)
    nsamples = vals_1d[0].shape[0]
    nterms = indices_np.shape[1]
    result: Array = bkd.asarray(
        basis_jacobian_numba(
            stacked_vals,
            stacked_derivs,
            indices_np,
            nvars,
            nsamples,
            nterms,
        )
    )
    return result


def _numba_basis_hessian(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    hess_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """Numba-backed basis Hessian."""
    from pyapprox.surrogates.affine.basis.compute_numba import (
        basis_hessian_numba,
    )

    stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
    stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
    stacked_hess = _stack_1d_arrays_for_numba(hess_1d)
    indices_np = np.asarray(indices)
    nsamples = vals_1d[0].shape[0]
    nterms = indices_np.shape[1]
    result: Array = bkd.asarray(
        basis_hessian_numba(
            stacked_vals,
            stacked_derivs,
            stacked_hess,
            indices_np,
            nvars,
            nsamples,
            nterms,
        )
    )
    return result


# --- torch.compile implementations ---


@lru_cache(maxsize=None)
def _get_compiled_basis_eval() -> Callable[[List[Array], Array], Array]:
    """Create and cache the torch.compile-wrapped basis eval kernel."""
    import torch

    from pyapprox.surrogates.affine.basis.compute_torch import (
        basis_eval_torch,
    )

    # cast: torch.compile preserves the Tensor signature (stub-version
    # dependent); the wrapper is used generically over Array
    return cast(
        Callable[[List[Array], Array], Array],
        torch.compile(basis_eval_torch),
    )


def _compiled_basis_eval(
    vals_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """torch.compile-backed basis evaluation."""
    return _get_compiled_basis_eval()(vals_1d, indices)


@lru_cache(maxsize=None)
def _get_compiled_basis_jacobian() -> (
    Callable[[List[Array], List[Array], Array], Array]
):
    """Create and cache the torch.compile-wrapped basis Jacobian kernel."""
    import torch

    from pyapprox.surrogates.affine.basis.compute_torch import (
        basis_jacobian_torch,
    )

    # cast: torch.compile preserves the Tensor signature (stub-version
    # dependent); the wrapper is used generically over Array
    return cast(
        Callable[[List[Array], List[Array], Array], Array],
        torch.compile(basis_jacobian_torch),
    )


def _compiled_basis_jacobian(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """torch.compile-backed basis Jacobian."""
    return _get_compiled_basis_jacobian()(vals_1d, derivs_1d, indices)


@lru_cache(maxsize=None)
def _get_compiled_basis_hessian() -> (
    Callable[[List[Array], List[Array], List[Array], Array], Array]
):
    """Create and cache the torch.compile-wrapped basis Hessian kernel."""
    import torch

    from pyapprox.surrogates.affine.basis.compute_torch import (
        basis_hessian_torch,
    )

    # cast: torch.compile preserves the Tensor signature (stub-version
    # dependent); the wrapper is used generically over Array
    return cast(
        Callable[[List[Array], List[Array], List[Array], Array], Array],
        torch.compile(basis_hessian_torch),
    )


def _compiled_basis_hessian(
    vals_1d: List[Array],
    derivs_1d: List[Array],
    hess_1d: List[Array],
    indices: Array,
    nvars: int,
    bkd: Backend[Array],
) -> Array:
    """torch.compile-backed basis Hessian."""
    return _get_compiled_basis_hessian()(vals_1d, derivs_1d, hess_1d, indices)


# --- Public dispatch functions ---


def get_basis_eval_impl(bkd: Backend[Array]) -> BasisEvalImpl[Array]:
    """Get the basis evaluation implementation for the given backend.

    Automatically selects the best implementation based on backend type.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (vals_1d, indices, nvars, bkd) -> Array
    """
    if _is_numpy(bkd) and _HAS_NUMBA:
        return _numba_basis_eval

    if _is_torch(bkd):
        return _compiled_basis_eval

    return basis_eval_vectorized


def get_basis_jacobian_impl(bkd: Backend[Array]) -> BasisJacobianImpl[Array]:
    """Get the basis Jacobian implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (vals_1d, derivs_1d, indices, nvars, bkd) -> Array
    """
    if _is_numpy(bkd) and _HAS_NUMBA:
        return _numba_basis_jacobian

    if _is_torch(bkd):
        return _compiled_basis_jacobian

    return basis_jacobian_vectorized


def get_basis_hessian_impl(bkd: Backend[Array]) -> BasisHessianImpl[Array]:
    """Get the basis Hessian implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (vals_1d, derivs_1d, hess_1d, indices, nvars, bkd) -> Array
    """
    if _is_numpy(bkd) and _HAS_NUMBA:
        return _numba_basis_hessian

    if _is_torch(bkd):
        return _compiled_basis_hessian

    return basis_hessian_vectorized
