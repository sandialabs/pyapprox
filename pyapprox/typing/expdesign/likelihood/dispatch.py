"""
Backend-aware dispatch for OED likelihood computations.

Selects between Numba fused kernels (for NumPy backend) and vectorized
backend-agnostic implementations. Each function returns a callable with
a uniform signature.

For NumPy backend with Numba available:
    - Uses fused kernels that avoid 3D intermediate arrays
    - Operates on raw NumPy arrays (zero-copy from NumpyBkd)

For all other backends (PyTorch, or NumPy without Numba):
    - Uses vectorized implementations from compute.py
    - Compatible with torch.compile for future acceleration
"""

from typing import Callable, Optional

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd

from pyapprox.typing.expdesign.likelihood.compute import (
    logpdf_matrix_vectorized,
    jacobian_matrix_vectorized,
    evidence_jacobian_vectorized,
)

try:
    from pyapprox.typing.expdesign.likelihood.compute_numba import (
        logpdf_matrix_numba,
        jacobian_matrix_numba,
        fused_evidence_jacobian_numba,
    )
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


LogpdfMatrixImpl = Callable[
    [Array, Array, Array, Array, Backend[Array]], Array
]

JacobianMatrixImpl = Callable[
    [Array, Array, Optional[Array], Array, Array, Backend[Array]], Array
]

EvidenceJacobianImpl = Callable[
    [
        Array,  # shapes
        Array,  # obs
        Optional[Array],  # latent_samples
        Array,  # base_variances
        Array,  # design_weights
        Array,  # quad_weighted_like
        Backend[Array],
    ],
    Array,
]


def _use_numba(bkd: Backend[Array], use_numba: bool) -> bool:
    """Check if Numba dispatch should be used."""
    return use_numba and HAS_NUMBA and isinstance(bkd, NumpyBkd)


def get_logpdf_matrix_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
) -> LogpdfMatrixImpl:
    """Get the logpdf_matrix implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, base_variances, design_weights, bkd) -> Array
    """
    if _use_numba(bkd, use_numba):
        def impl(
            shapes: Array,
            obs: Array,
            base_variances: Array,
            design_weights: Array,
            bkd: Backend[Array],
        ) -> Array:
            return logpdf_matrix_numba(
                shapes, obs, base_variances, design_weights,
            )
        return impl
    return logpdf_matrix_vectorized


def get_jacobian_matrix_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
) -> JacobianMatrixImpl:
    """Get the jacobian_matrix implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, latent_samples, base_variances, design_weights, bkd) -> Array
    """
    if _use_numba(bkd, use_numba):
        def impl(
            shapes: Array,
            obs: Array,
            latent_samples: Optional[Array],
            base_variances: Array,
            design_weights: Array,
            bkd: Backend[Array],
        ) -> Array:
            has_latent = latent_samples is not None
            if not has_latent:
                # Numba needs a concrete array; pass zeros (ignored)
                latent_samples_np = np.zeros_like(obs)
            else:
                latent_samples_np = latent_samples
            return jacobian_matrix_numba(
                shapes, obs, latent_samples_np,
                base_variances, design_weights, has_latent,
            )
        return impl
    return jacobian_matrix_vectorized


def get_evidence_jacobian_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
) -> EvidenceJacobianImpl:
    """Get the fused evidence jacobian implementation for the given backend.

    For NumPy+Numba: uses the fused kernel that avoids 3D materialization.
    For other backends: computes jacobian_matrix + einsum separately.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, latent_samples, base_variances, design_weights,
         quad_weighted_like, bkd) -> Array
    """
    if _use_numba(bkd, use_numba):
        def impl(
            shapes: Array,
            obs: Array,
            latent_samples: Optional[Array],
            base_variances: Array,
            design_weights: Array,
            quad_weighted_like: Array,
            bkd: Backend[Array],
        ) -> Array:
            has_latent = latent_samples is not None
            if not has_latent:
                latent_samples_np = np.zeros_like(obs)
            else:
                latent_samples_np = latent_samples
            return fused_evidence_jacobian_numba(
                shapes, obs, latent_samples_np,
                base_variances, design_weights,
                quad_weighted_like, has_latent,
            )
        return impl

    def vectorized_impl(
        shapes: Array,
        obs: Array,
        latent_samples: Optional[Array],
        base_variances: Array,
        design_weights: Array,
        quad_weighted_like: Array,
        bkd: Backend[Array],
    ) -> Array:
        loglike_jac = jacobian_matrix_vectorized(
            shapes, obs, latent_samples, base_variances, design_weights, bkd,
        )
        return evidence_jacobian_vectorized(loglike_jac, quad_weighted_like, bkd)
    return vectorized_impl
