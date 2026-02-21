"""
Backend-aware dispatch for MultiIndexBasis tensor product assembly.

Selects between three acceleration strategies based on the backend type:
1. Numba fused kernels (for NumPy backend) — avoids all intermediate arrays
2. torch-native functions (for PyTorch backend)
3. Vectorized backend-agnostic implementations (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so that
MultiIndexBasis is unaware of which strategy is active.
"""

from typing import Callable, List

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")

from pyapprox.typing.surrogates.affine.basis.compute import (
    basis_eval_vectorized,
    basis_jacobian_vectorized,
    basis_hessian_vectorized,
)

# Type aliases for dispatch callables
BasisEvalImpl = Callable[
    [List[Array], Array, int, Backend[Array]], Array
]

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
    from pyapprox.typing.util.backends.torch import TorchBkd
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
        stacked[dd, :, :vals_1d[dd].shape[1]] = vals_1d[dd]
    return stacked


# --- Public dispatch functions ---

def get_basis_eval_impl(bkd: Backend[Array]) -> BasisEvalImpl:
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
        from pyapprox.typing.surrogates.affine.basis.compute_numba import (
            basis_eval_numba,
        )

        def impl(
            vals_1d: List[Array],
            indices: Array,
            nvars: int,
            bkd: Backend[Array],
        ) -> Array:
            stacked = _stack_1d_arrays_for_numba(vals_1d)
            indices_np = np.asarray(indices)
            nsamples = vals_1d[0].shape[0]
            nterms = indices_np.shape[1]
            return basis_eval_numba(
                stacked, indices_np, nvars, nsamples, nterms,
            )
        return impl

    if _is_torch(bkd):
        return _make_compiled_eval()

    return basis_eval_vectorized


def get_basis_jacobian_impl(bkd: Backend[Array]) -> BasisJacobianImpl:
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
        from pyapprox.typing.surrogates.affine.basis.compute_numba import (
            basis_jacobian_numba,
        )

        def impl(
            vals_1d: List[Array],
            derivs_1d: List[Array],
            indices: Array,
            nvars: int,
            bkd: Backend[Array],
        ) -> Array:
            stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
            stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
            indices_np = np.asarray(indices)
            nsamples = vals_1d[0].shape[0]
            nterms = indices_np.shape[1]
            return basis_jacobian_numba(
                stacked_vals, stacked_derivs, indices_np,
                nvars, nsamples, nterms,
            )
        return impl

    if _is_torch(bkd):
        return _make_compiled_jacobian()

    return basis_jacobian_vectorized


def get_basis_hessian_impl(bkd: Backend[Array]) -> BasisHessianImpl:
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
        from pyapprox.typing.surrogates.affine.basis.compute_numba import (
            basis_hessian_numba,
        )

        def impl(
            vals_1d: List[Array],
            derivs_1d: List[Array],
            hess_1d: List[Array],
            indices: Array,
            nvars: int,
            bkd: Backend[Array],
        ) -> Array:
            stacked_vals = _stack_1d_arrays_for_numba(vals_1d)
            stacked_derivs = _stack_1d_arrays_for_numba(derivs_1d)
            stacked_hess = _stack_1d_arrays_for_numba(hess_1d)
            indices_np = np.asarray(indices)
            nsamples = vals_1d[0].shape[0]
            nterms = indices_np.shape[1]
            return basis_hessian_numba(
                stacked_vals, stacked_derivs, stacked_hess, indices_np,
                nvars, nsamples, nterms,
            )
        return impl

    if _is_torch(bkd):
        return _make_compiled_hessian()

    return basis_hessian_vectorized


# --- torch.compile wrapper factories ---

def _make_compiled_eval() -> BasisEvalImpl:
    """Create a torch.compile-wrapped basis eval implementation."""
    import torch
    from pyapprox.typing.surrogates.affine.basis.compute_torch import (
        basis_eval_torch,
    )
    compiled_fn = torch.compile(basis_eval_torch)

    def impl(
        vals_1d: List[Array],
        indices: Array,
        nvars: int,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(vals_1d, indices)
    return impl


def _make_compiled_jacobian() -> BasisJacobianImpl:
    """Create a torch.compile-wrapped basis Jacobian implementation."""
    import torch
    from pyapprox.typing.surrogates.affine.basis.compute_torch import (
        basis_jacobian_torch,
    )
    compiled_fn = torch.compile(basis_jacobian_torch)

    def impl(
        vals_1d: List[Array],
        derivs_1d: List[Array],
        indices: Array,
        nvars: int,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(vals_1d, derivs_1d, indices)
    return impl


def _make_compiled_hessian() -> BasisHessianImpl:
    """Create a torch.compile-wrapped basis Hessian implementation."""
    import torch
    from pyapprox.typing.surrogates.affine.basis.compute_torch import (
        basis_hessian_torch,
    )
    compiled_fn = torch.compile(basis_hessian_torch)

    def impl(
        vals_1d: List[Array],
        derivs_1d: List[Array],
        hess_1d: List[Array],
        indices: Array,
        nvars: int,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(vals_1d, derivs_1d, hess_1d, indices)
    return impl
