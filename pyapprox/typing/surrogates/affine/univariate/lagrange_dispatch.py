"""Backend-aware dispatch for Lagrange polynomial evaluation.

Selects the best acceleration strategy based on the backend type:
1. Numba kernel (NumpyBkd) — parallel barycentric evaluation in C
2. torch.compile-wrapped torch-native implementation (TorchBkd)
3. Backend-generic barycentric formula (fallback) — uses bkd.* methods
"""

from typing import Callable, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd


# Uniform signature: (abscissa, samples, bary_weights, bkd) -> values
LagrangeEvalImpl = Callable[
    [Array, Array, Array, Backend[Array]], Array
]


def _make_numba_lagrange_eval() -> LagrangeEvalImpl:
    """Create a numba-backed Lagrange evaluation implementation."""
    from pyapprox.typing.surrogates.affine.univariate.lagrange_numba import (
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
    from pyapprox.typing.surrogates.affine.univariate.lagrange_torch import (
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
    if isinstance(bkd, NumpyBkd):
        return _make_numba_lagrange_eval()
    if isinstance(bkd, TorchBkd):
        return _make_compiled_lagrange_eval()
    return _generic_lagrange_eval
