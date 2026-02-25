"""Backend-aware dispatch for Lagrange polynomial evaluation and derivatives.

Selects the best acceleration strategy based on the backend type:
1. Numba kernel (NumpyBkd) — parallel barycentric evaluation in C
2. torch.compile-wrapped torch-native implementation (TorchBkd)
3. Backend-generic barycentric formula (fallback) — uses bkd.* methods
"""

from typing import Callable, Optional

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")


# Uniform signature: (abscissa, samples, bary_weights, bkd) -> values
LagrangeEvalImpl = Callable[
    [Array, Array, Array, Backend[Array]], Array
]

# Uniform signature for derivatives: same as eval
LagrangeDerivImpl = Callable[
    [Array, Array, Array, Backend[Array]], Array
]


def _is_torch(bkd: Backend[Array]) -> bool:
    """Check if backend is PyTorch; import deferred to avoid torch load time."""
    from pyapprox.util.backends.torch import TorchBkd
    return isinstance(bkd, TorchBkd)


# ---------------------------------------------------------------------------
# Evaluation implementations
# ---------------------------------------------------------------------------

def _make_numba_lagrange_eval() -> LagrangeEvalImpl:
    """Create a numba-backed Lagrange evaluation implementation."""
    from pyapprox.surrogates.affine.univariate.lagrange_numba import (
        lagrange_eval_numba,
    )

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return lagrange_eval_numba(
            np.asarray(abscissa),
            np.asarray(samples),
            np.asarray(bary_weights),
        )

    return impl


def _generic_lagrange_eval(
    abscissa: Array,
    samples: Array,
    bary_weights: Array,
    bkd: Backend[Array],
) -> Array:
    """Evaluate Lagrange basis using vectorized barycentric formula.

    Backend-generic fallback that uses bkd.* methods. Works with any
    backend including PyTorch (preserves autograd computation graph).

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes, shape (nabscissa,).
    samples : Array
        Evaluation points, shape (nsamples,).
    bary_weights : Array
        Precomputed barycentric weights, shape (nabscissa,).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Basis values, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    if nabscissa == 1:
        return bkd.ones((samples.shape[0], 1))

    # Differences: (nsamples, nabscissa)
    numers = samples[:, None] - abscissa[None, :]

    # Detect exact node hits
    is_node = numers == 0.0

    # Replace zeros with 1.0 for safe product computation
    safe_numers = bkd.where(is_node, 1.0, numers)

    # P(x) with safe numerators: (nsamples, 1)
    full_product = bkd.prod(safe_numers, axis=1, keepdims=True)

    # L_j(x) = w_j * P(x) / (x - x_j)
    values = bary_weights[None, :] * full_product / safe_numers

    # Override rows where sample coincides with a node
    any_hit = bkd.any_array(is_node, axis=1)  # (nsamples,)
    any_hit_2d = bkd.reshape(any_hit, (-1, 1))  # (nsamples, 1)
    node_values = bkd.where(is_node, 1.0, 0.0)
    values = bkd.where(any_hit_2d, node_values, values)

    return values


def _make_compiled_lagrange_eval() -> LagrangeEvalImpl:
    """Create a torch.compile-wrapped Lagrange evaluation implementation."""
    import torch
    from pyapprox.surrogates.affine.univariate.lagrange_torch import (
        lagrange_eval_torch,
    )

    compiled_fn = torch.compile(lagrange_eval_torch)

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(abscissa, samples, bary_weights)

    return impl


def get_lagrange_eval_impl(bkd: Backend[Array]) -> LagrangeEvalImpl:
    """Get the Lagrange evaluation implementation for the given backend.

    Automatically selects the best strategy based on the backend type:
    - NumpyBkd: Numba parallel kernel
    - TorchBkd: torch.compile-wrapped torch-native implementation
    - All other backends: vectorized backend-generic fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (abscissa, samples, bary_weights, bkd) -> Array
    """
    if isinstance(bkd, NumpyBkd) and _HAS_NUMBA:
        return _make_numba_lagrange_eval()
    if _is_torch(bkd):
        return _make_compiled_lagrange_eval()
    return _generic_lagrange_eval


# ---------------------------------------------------------------------------
# First derivative implementations
# ---------------------------------------------------------------------------

def _generic_lagrange_jacobian(
    abscissa: Array,
    samples: Array,
    bary_weights: Array,
    bkd: Backend[Array],
) -> Array:
    """Evaluate first derivatives of Lagrange basis polynomials.

    Uses the identity L'_j(x) = L_j(x) * S_j(x) where
    S_j(x) = sum_{k!=j} 1/(x - x_k).

    At near-nodes x ≈ x_m, uses the closed-form limit:
    - L'_j(x_m) = w_j / (w_m * (x_m - x_j))  for j != m
    - L'_m(x_m) = -sum_{k!=m} L'_k(x_m)       (partition of unity)

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes, shape (nabscissa,).
    samples : Array
        Evaluation points, shape (nsamples,).
    bary_weights : Array
        Precomputed barycentric weights, shape (nabscissa,).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        First derivatives, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    nsamples = samples.shape[0]
    if nabscissa == 1:
        return bkd.zeros((nsamples, 1))

    # diffs[i, k] = x_i - x_k, shape (nsamples, nabscissa)
    diffs = samples[:, None] - abscissa[None, :]

    # Near-node detection with tolerance (derivatives use 1/(x-x_k)
    # which suffers catastrophic cancellation when x ≈ x_k)
    abs_diffs = bkd.abs(diffs)
    tol = 1e-12 * (1.0 + bkd.max(bkd.abs(abscissa)))
    is_near_node = abs_diffs < tol

    # Safe diffs for division (replace near-zeros with 1.0)
    safe_diffs = bkd.where(is_near_node, 1.0, diffs)

    # L_j(x) via barycentric formula
    basis_vals = _generic_lagrange_eval(abscissa, samples, bary_weights, bkd)

    # S_j(x) = sum_{k!=j} 1/(x - x_k)
    inv_diffs = 1.0 / safe_diffs
    inv_diffs = bkd.where(is_near_node, 0.0, inv_diffs)

    total_sum = bkd.sum(inv_diffs, axis=1, keepdims=True)
    S = total_sum - inv_diffs

    # Regular case: L'_j(x) = L_j(x) * S_j(x)
    derivs = basis_vals * S

    # Node case: precompute D1[m, j] = L'_j(x_m)
    node_diffs = abscissa[:, None] - abscissa[None, :]
    node_is_diag = node_diffs == 0.0
    safe_node_diffs = bkd.where(node_is_diag, 1.0, node_diffs)
    node_derivs = bary_weights[None, :] / (
        bary_weights[:, None] * safe_node_diffs
    )
    off_diag = bkd.where(node_is_diag, 0.0, node_derivs)
    diag_vals = -bkd.sum(off_diag, axis=1)
    node_derivs = bkd.where(node_is_diag, diag_vals[:, None], off_diag)

    # For near-node samples, look up the precomputed row via matmul
    any_hit = bkd.any_array(is_near_node, axis=1)
    any_hit_2d = bkd.reshape(any_hit, (-1, 1))
    is_near_float = bkd.where(is_near_node, 1.0, 0.0)
    node_result = bkd.dot(is_near_float, node_derivs)

    derivs = bkd.where(any_hit_2d, node_result, derivs)

    return derivs


def _make_numba_lagrange_jacobian() -> LagrangeDerivImpl:
    """Create a numba-backed first derivative implementation."""
    from pyapprox.surrogates.affine.univariate.lagrange_numba import (
        lagrange_jacobian_numba,
    )

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return lagrange_jacobian_numba(
            np.asarray(abscissa),
            np.asarray(samples),
            np.asarray(bary_weights),
        )

    return impl


def _make_compiled_lagrange_jacobian() -> LagrangeDerivImpl:
    """Create a torch.compile-wrapped first derivative implementation."""
    import torch
    from pyapprox.surrogates.affine.univariate.lagrange_torch import (
        lagrange_jacobian_torch,
    )

    compiled_fn = torch.compile(lagrange_jacobian_torch)

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(abscissa, samples, bary_weights)

    return impl


def get_lagrange_jacobian_impl(bkd: Backend[Array]) -> LagrangeDerivImpl:
    """Get the first derivative implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (abscissa, samples, bary_weights, bkd) -> Array
    """
    if isinstance(bkd, NumpyBkd) and _HAS_NUMBA:
        return _make_numba_lagrange_jacobian()
    if _is_torch(bkd):
        return _make_compiled_lagrange_jacobian()
    return _generic_lagrange_jacobian


# ---------------------------------------------------------------------------
# Second derivative implementations
# ---------------------------------------------------------------------------

def _generic_lagrange_hessian(
    abscissa: Array,
    samples: Array,
    bary_weights: Array,
    bkd: Backend[Array],
) -> Array:
    """Evaluate second derivatives of Lagrange basis polynomials.

    Uses L''_j(x) = L_j(x) * [S_j(x)^2 - T_j(x)] where
    S_j(x) = sum_{k!=j} 1/(x - x_k), T_j(x) = sum_{k!=j} 1/(x - x_k)^2.

    At near-nodes x ≈ x_m, uses closed-form limits.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes, shape (nabscissa,).
    samples : Array
        Evaluation points, shape (nsamples,).
    bary_weights : Array
        Precomputed barycentric weights, shape (nabscissa,).
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Second derivatives, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    nsamples = samples.shape[0]
    if nabscissa == 1:
        return bkd.zeros((nsamples, 1))

    # diffs[i, k] = x_i - x_k, shape (nsamples, nabscissa)
    diffs = samples[:, None] - abscissa[None, :]

    # Near-node detection with tolerance
    abs_diffs = bkd.abs(diffs)
    tol = 1e-12 * (1.0 + bkd.max(bkd.abs(abscissa)))
    is_near_node = abs_diffs < tol

    safe_diffs = bkd.where(is_near_node, 1.0, diffs)

    # L_j(x) via barycentric formula
    basis_vals = _generic_lagrange_eval(abscissa, samples, bary_weights, bkd)

    inv_diffs = 1.0 / safe_diffs
    inv_diffs = bkd.where(is_near_node, 0.0, inv_diffs)

    # S_j(x) = sum_{k!=j} 1/(x - x_k)
    total_sum = bkd.sum(inv_diffs, axis=1, keepdims=True)
    S = total_sum - inv_diffs

    # T_j(x) = sum_{k!=j} 1/(x - x_k)^2
    inv_diffs_sq = inv_diffs * inv_diffs
    total_sum_sq = bkd.sum(inv_diffs_sq, axis=1, keepdims=True)
    T = total_sum_sq - inv_diffs_sq

    # Regular case: L''_j(x) = L_j(x) * (S_j(x)^2 - T_j(x))
    derivs = basis_vals * (S * S - T)

    # Node case: precompute D1 and D2 matrices
    node_diffs_mat = abscissa[:, None] - abscissa[None, :]
    node_is_diag = node_diffs_mat == 0.0
    safe_node_diffs = bkd.where(node_is_diag, 1.0, node_diffs_mat)

    D1 = bary_weights[None, :] / (bary_weights[:, None] * safe_node_diffs)
    D1_off = bkd.where(node_is_diag, 0.0, D1)
    D1_diag = -bkd.sum(D1_off, axis=1)
    D1 = bkd.where(node_is_diag, D1_diag[:, None], D1_off)

    inv_node_diffs = 1.0 / safe_node_diffs
    inv_node_diffs = bkd.where(node_is_diag, 0.0, inv_node_diffs)
    D2_off = 2.0 * D1_off * (D1_diag[:, None] - inv_node_diffs)
    D2_diag = -bkd.sum(D2_off, axis=1)
    D2 = bkd.where(node_is_diag, D2_diag[:, None], D2_off)

    any_hit = bkd.any_array(is_near_node, axis=1)
    any_hit_2d = bkd.reshape(any_hit, (-1, 1))
    is_near_float = bkd.where(is_near_node, 1.0, 0.0)
    node_result = bkd.dot(is_near_float, D2)

    derivs = bkd.where(any_hit_2d, node_result, derivs)

    return derivs


def _make_numba_lagrange_hessian() -> LagrangeDerivImpl:
    """Create a numba-backed second derivative implementation."""
    from pyapprox.surrogates.affine.univariate.lagrange_numba import (
        lagrange_hessian_numba,
    )

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return lagrange_hessian_numba(
            np.asarray(abscissa),
            np.asarray(samples),
            np.asarray(bary_weights),
        )

    return impl


def _make_compiled_lagrange_hessian() -> LagrangeDerivImpl:
    """Create a torch.compile-wrapped second derivative implementation."""
    import torch
    from pyapprox.surrogates.affine.univariate.lagrange_torch import (
        lagrange_hessian_torch,
    )

    compiled_fn = torch.compile(lagrange_hessian_torch)

    def impl(
        abscissa: Array,
        samples: Array,
        bary_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(abscissa, samples, bary_weights)

    return impl


def get_lagrange_hessian_impl(bkd: Backend[Array]) -> LagrangeDerivImpl:
    """Get the second derivative implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (abscissa, samples, bary_weights, bkd) -> Array
    """
    if isinstance(bkd, NumpyBkd) and _HAS_NUMBA:
        return _make_numba_lagrange_hessian()
    if _is_torch(bkd):
        return _make_compiled_lagrange_hessian()
    return _generic_lagrange_hessian
