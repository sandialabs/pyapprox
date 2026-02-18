"""
Backend-aware dispatch for OED likelihood computations.

Selects between three acceleration strategies:
1. Numba fused kernels (for NumPy backend) — avoids 3D intermediate arrays
2. torch.compile-wrapped torch-native functions (for PyTorch backend)
3. Vectorized backend-agnostic implementations (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so that
the calling class (GaussianOEDInnerLoopLikelihood) is unaware of which
strategy is active.

TODO: Refactor to match tensorproduct/dispatch.py pattern:
  - Remove use_numba/use_torch_compile flags; auto-dispatch from bkd type
  - Make numba/torch top-level imports (both are required dependencies)
  - Remove try/except guards for numba and TorchBkd imports
  - Update gaussian.py __init__ and tests accordingly
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

from pyapprox.typing.util.optional_deps import package_available

HAS_NUMBA = package_available("numba")
if HAS_NUMBA:
    from pyapprox.typing.expdesign.likelihood.compute_numba import (
        logpdf_matrix_numba,
        jacobian_matrix_numba,
        fused_evidence_jacobian_numba,
    )


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


def _use_torch_compile(bkd: Backend[Array], use_torch_compile: bool) -> bool:
    """Check if torch.compile dispatch should be used."""
    if not use_torch_compile:
        return False
    try:
        from pyapprox.typing.util.backends.torch import TorchBkd
        return isinstance(bkd, TorchBkd)
    except ImportError:
        return False


# --- torch.compile wrapper factories ---

def _make_compiled_logpdf() -> LogpdfMatrixImpl:
    """Create a torch.compile-wrapped logpdf_matrix implementation."""
    import torch
    from pyapprox.typing.expdesign.likelihood.compute_torch import (
        logpdf_matrix_torch,
    )

    compiled_fn = torch.compile(logpdf_matrix_torch)

    def impl(
        shapes: Array,
        obs: Array,
        base_variances: Array,
        design_weights: Array,
        bkd: Backend[Array],
    ) -> Array:
        return compiled_fn(shapes, obs, base_variances, design_weights)

    return impl


def _make_compiled_jacobian() -> JacobianMatrixImpl:
    """Create a torch.compile-wrapped jacobian_matrix implementation."""
    import torch
    from pyapprox.typing.expdesign.likelihood.compute_torch import (
        jacobian_matrix_torch,
    )

    compiled_fn = torch.compile(jacobian_matrix_torch)

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
            import torch as _torch
            latent_samples_t = _torch.zeros_like(obs)
        else:
            latent_samples_t = latent_samples
        return compiled_fn(
            shapes, obs, latent_samples_t,
            base_variances, design_weights, has_latent,
        )

    return impl


def _make_compiled_evidence_jacobian() -> EvidenceJacobianImpl:
    """Create a torch.compile-wrapped evidence jacobian implementation."""
    import torch
    from pyapprox.typing.expdesign.likelihood.compute_torch import (
        jacobian_matrix_torch,
        evidence_jacobian_torch,
    )

    compiled_jac = torch.compile(jacobian_matrix_torch)
    compiled_ev_jac = torch.compile(evidence_jacobian_torch)

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
            import torch as _torch
            latent_samples_t = _torch.zeros_like(obs)
        else:
            latent_samples_t = latent_samples
        loglike_jac = compiled_jac(
            shapes, obs, latent_samples_t,
            base_variances, design_weights, has_latent,
        )
        return compiled_ev_jac(loglike_jac, quad_weighted_like)

    return impl


# --- Public dispatch functions ---

def get_logpdf_matrix_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
    use_torch_compile: bool = False,
) -> LogpdfMatrixImpl:
    """Get the logpdf_matrix implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.
    use_torch_compile : bool
        Whether to use torch.compile if available. Default False.

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
    if _use_torch_compile(bkd, use_torch_compile):
        return _make_compiled_logpdf()
    return logpdf_matrix_vectorized


def get_jacobian_matrix_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
    use_torch_compile: bool = False,
) -> JacobianMatrixImpl:
    """Get the jacobian_matrix implementation for the given backend.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.
    use_torch_compile : bool
        Whether to use torch.compile if available. Default False.

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
    if _use_torch_compile(bkd, use_torch_compile):
        return _make_compiled_jacobian()
    return jacobian_matrix_vectorized


def get_evidence_jacobian_impl(
    bkd: Backend[Array],
    use_numba: bool = True,
    use_torch_compile: bool = False,
) -> EvidenceJacobianImpl:
    """Get the fused evidence jacobian implementation for the given backend.

    For NumPy+Numba: uses the fused kernel that avoids 3D materialization.
    For Torch+compile: uses compiled jacobian + compiled einsum.
    For other backends: computes jacobian_matrix + einsum separately.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    use_numba : bool
        Whether to use Numba if available. Default True.
    use_torch_compile : bool
        Whether to use torch.compile if available. Default False.

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

    if _use_torch_compile(bkd, use_torch_compile):
        return _make_compiled_evidence_jacobian()

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
