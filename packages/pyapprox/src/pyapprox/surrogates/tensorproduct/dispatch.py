"""Backend-aware dispatch for tensor product evaluation.

Selects the best acceleration strategy based on the backend type:
1. Numba contraction kernel (NumpyBkd + numba available)
2. torch.compile-wrapped torch-native einsum (TorchBkd)
3. Vectorized backend-agnostic einsum (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so that
the TensorProductInterpolant is unaware of which strategy is active.

All dispatched implementations are module-level functions (not closures) so
that objects storing them as attributes remain picklable.
"""

from functools import lru_cache
from typing import Callable, List, cast

import numpy as np

from pyapprox.surrogates.tensorproduct.compute import (
    tp_eval_vectorized,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

_HAS_NUMBA = package_available("numba")


def _is_torch(bkd: Backend[Array]) -> bool:
    """Check if backend is PyTorch; import deferred to avoid torch load time."""
    from pyapprox.util.backends.torch import TorchBkd

    return isinstance(bkd, TorchBkd)


TpEvalImpl = Callable[[List[Array], Array, List[int], Backend[Array]], Array]


def _numba_tp_eval(
    basis_vals_1d: List[Array],
    values: Array,
    nterms_1d: List[int],
    bkd: Backend[Array],
) -> Array:
    """Numba-backed tp_eval implementation.

    Wraps the raw Numba kernel by converting List[Array] to padded
    (nvars, npoints, max_n1d) array before calling the kernel.
    """
    from pyapprox.surrogates.tensorproduct.compute_numba import (
        tp_eval_numba,
    )

    nvars = len(nterms_1d)
    nqoi = values.shape[0]
    npoints = basis_vals_1d[0].shape[0]
    max_n1d = max(nterms_1d)

    # Pack into padded array (nvars, npoints, max_n1d)
    basis_vals_pad = np.zeros((nvars, npoints, max_n1d))
    for d in range(nvars):
        n_d = nterms_1d[d]
        basis_vals_pad[d, :, :n_d] = basis_vals_1d[d]

    nterms_1d_arr = np.array(nterms_1d, dtype=np.int64)

    result: Array = bkd.asarray(
        tp_eval_numba(
            np.asarray(values),
            basis_vals_pad,
            nterms_1d_arr,
            nvars,
            nqoi,
            npoints,
        )
    )
    return result


@lru_cache(maxsize=None)
def _get_compiled_tp_eval() -> Callable[[List[Array], Array, List[int]], Array]:
    """Create and cache the torch.compile-wrapped tp_eval kernel."""
    import torch

    from pyapprox.surrogates.tensorproduct.compute_torch import (
        tp_eval_torch,
    )

    # cast: torch.compile preserves the Tensor signature (stub-version
    # dependent); the wrapper is used generically over Array
    return cast(
        Callable[[List[Array], Array, List[int]], Array],
        torch.compile(tp_eval_torch),
    )


def _compiled_tp_eval(
    basis_vals_1d: List[Array],
    values: Array,
    nterms_1d: List[int],
    bkd: Backend[Array],
) -> Array:
    """torch.compile-backed tp_eval implementation.

    Uses torch.einsum directly (bypassing bkd.*) to avoid graph breaks
    during torch.compile tracing.
    """
    return _get_compiled_tp_eval()(basis_vals_1d, values, nterms_1d)


def get_tp_eval_impl(bkd: Backend[Array]) -> TpEvalImpl[Array]:
    """Get the tensor product evaluation implementation for the given backend.

    Automatically selects the best strategy based on the backend type:
    - NumpyBkd with Numba available: Numba contraction kernel
    - TorchBkd: torch.compile-wrapped torch-native einsum
    - All other backends: vectorized einsum fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (basis_vals_1d, values, nterms_1d, bkd) -> Array
    """
    if isinstance(bkd, NumpyBkd) and _HAS_NUMBA:
        return _numba_tp_eval
    if _is_torch(bkd):
        return _compiled_tp_eval
    return tp_eval_vectorized
