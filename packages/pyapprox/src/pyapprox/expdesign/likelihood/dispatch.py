"""
Backend-aware dispatch for OED likelihood computations.

Selects between three acceleration strategies based on backend type:
1. Numba fused kernels (for NumPy backend when numba is installed)
2. torch.compile-wrapped torch-native functions (for PyTorch backend)
3. Vectorized backend-agnostic implementations (fallback for any backend)

Each dispatch function returns a callable with a uniform signature so that
the calling class (GaussianOEDInnerLoopLikelihood) is unaware of which
strategy is active. Dispatch is fully automatic — no flags needed.
"""

from typing import Callable, Optional, Tuple

import numpy as np

from pyapprox.expdesign.likelihood.compute import (
    evidence_jacobian_vectorized,
    jacobian_matrix_vectorized,
    logpdf_matrix_vectorized,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.optional_deps import package_available

HAS_NUMBA = package_available("numba")
if HAS_NUMBA:
    from pyapprox.expdesign.likelihood.compute_numba import (
        fused_evidence_jacobian_numba,
        fused_weighted_jacobian_numba,
        jacobian_matrix_numba,
        logpdf_matrix_numba,
    )


LogpdfMatrixImpl = Callable[[Array, Array, Array, Array, Backend[Array]], Array]

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

WeightedJacobianImpl = Callable[
    [
        Array,            # shapes         (nobs, ninner)
        Array,            # obs            (nobs, nouter)
        Optional[Array],  # latent_samples (nobs, nouter)
        Array,            # base_variances (nobs,)
        Array,            # design_weights (nobs, 1)
        Array,            # qwl_ratio      (ninner, nouter) = qwl/evid
        Array,            # weights_a      (ninner, npred)
        Array,            # weights_b      (ninner, npred)
        Backend[Array],
    ],
    Tuple[Array, Array],  # (part_a, part_b), each (npred, nouter, nobs)
]


def _is_numpy(bkd: Backend[Array]) -> bool:
    """Check if the backend is NumPy."""
    return isinstance(bkd, NumpyBkd)


def _is_torch(bkd: Backend[Array]) -> bool:
    """Check if the backend is PyTorch."""
    try:
        from pyapprox.util.backends.torch import TorchBkd

        return isinstance(bkd, TorchBkd)
    except ImportError:
        return False


# --- torch.compile wrapper factories ---


def _check_torch_compile_available() -> None:
    """Verify torch.compile can work (C++ compiler available).

    Raises
    ------
    RuntimeError
        If no working C++ compiler is found for torch inductor.
    """
    import subprocess

    for cxx in ("clang++", "g++"):
        try:
            subprocess.check_output(
                [cxx, "--version"], stderr=subprocess.DEVNULL
            )
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    raise RuntimeError(
        "torch.compile requires a working C++ compiler (clang++ or g++) "
        "but none was found. On macOS, run 'sudo xcodebuild -license' "
        "and install Xcode Command Line Tools. On Linux, install g++ "
        "(e.g. 'apt install g++' or 'conda install -c conda-forge gxx')."
    )


def _make_compiled_logpdf() -> LogpdfMatrixImpl[Array]:
    """Create a torch.compile-wrapped logpdf_matrix implementation."""
    import torch

    _check_torch_compile_available()
    from pyapprox.expdesign.likelihood.compute_torch import (
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


def _make_compiled_jacobian() -> JacobianMatrixImpl[Array]:
    """Create a torch.compile-wrapped jacobian_matrix implementation."""
    import torch

    _check_torch_compile_available()
    from pyapprox.expdesign.likelihood.compute_torch import (
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
            shapes,
            obs,
            latent_samples_t,
            base_variances,
            design_weights,
            has_latent,
        )

    return impl


def _make_compiled_evidence_jacobian() -> EvidenceJacobianImpl[Array]:
    """Create a torch.compile-wrapped evidence jacobian implementation."""
    import torch

    _check_torch_compile_available()
    from pyapprox.expdesign.likelihood.compute_torch import (
        evidence_jacobian_torch,
        jacobian_matrix_torch,
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
            shapes,
            obs,
            latent_samples_t,
            base_variances,
            design_weights,
            has_latent,
        )
        return compiled_ev_jac(loglike_jac, quad_weighted_like)

    return impl


# --- Public dispatch functions ---


def get_logpdf_matrix_impl(
    bkd: Backend[Array],
) -> LogpdfMatrixImpl[Array]:
    """Get the logpdf_matrix implementation for the given backend.

    Dispatch order:
    1. NumPy + Numba installed → Numba fused kernel
    2. PyTorch → torch.compile wrapper
    3. Otherwise → vectorized fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, base_variances, design_weights, bkd) -> Array
    """
    if _is_numpy(bkd) and HAS_NUMBA:

        def impl(
            shapes: Array,
            obs: Array,
            base_variances: Array,
            design_weights: Array,
            bkd: Backend[Array],
        ) -> Array:
            return logpdf_matrix_numba(
                shapes,
                obs,
                base_variances,
                design_weights,
            )

        return impl
    if _is_torch(bkd):
        return _make_compiled_logpdf()
    return logpdf_matrix_vectorized


def get_jacobian_matrix_impl(
    bkd: Backend[Array],
) -> JacobianMatrixImpl[Array]:
    """Get the jacobian_matrix implementation for the given backend.

    Dispatch order:
    1. NumPy + Numba installed → Numba fused kernel
    2. PyTorch → torch.compile wrapper
    3. Otherwise → vectorized fallback

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, latent_samples, base_variances, design_weights, bkd) -> Array
    """
    if _is_numpy(bkd) and HAS_NUMBA:

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
                shapes,
                obs,
                latent_samples_np,
                base_variances,
                design_weights,
                has_latent,
            )

        return impl
    if _is_torch(bkd):
        return _make_compiled_jacobian()
    return jacobian_matrix_vectorized


def get_evidence_jacobian_impl(
    bkd: Backend[Array],
) -> EvidenceJacobianImpl[Array]:
    """Get the fused evidence jacobian implementation for the given backend.

    For NumPy+Numba: uses the fused kernel that avoids 3D materialization.
    For Torch+compile: uses compiled jacobian + compiled einsum.
    For other backends: computes jacobian_matrix + einsum separately.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable
        Implementation with signature:
        (shapes, obs, latent_samples, base_variances, design_weights,
         quad_weighted_like, bkd) -> Array
    """
    if _is_numpy(bkd) and HAS_NUMBA:

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
                shapes,
                obs,
                latent_samples_np,
                base_variances,
                design_weights,
                quad_weighted_like,
                has_latent,
            )

        return impl

    if _is_torch(bkd):
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
            shapes,
            obs,
            latent_samples,
            base_variances,
            design_weights,
            bkd,
        )
        return evidence_jacobian_vectorized(loglike_jac, quad_weighted_like, bkd)

    return vectorized_impl


def get_weighted_jacobian_impl(
    bkd: Backend[Array],
) -> Optional[WeightedJacobianImpl[Array]]:
    """Get the fused weighted-jacobian implementation, or None.

    Returns an impl that computes, for two independent per-inner weight
    matrices ``weights_a[i, q]`` and ``weights_b[i, q]``:

        part_a[q, j, k] = sum_i weights_a[i, q] * qwl_ratio[i, j] * Jlog[i, j, k]
        part_b[q, j, k] = sum_i weights_b[i, q] * qwl_ratio[i, j] * Jlog[i, j, k]

    This avoids materializing the ``(ninner, nouter, nobs)``
    loglike-jacobian intermediate and replaces two ``"iq,iod->qod"``
    einsums in the deviation-measure jacobian with a per-thread dgemm.

    Used by ``StandardDeviationMeasure`` (with ``weights_a = qoi``,
    ``weights_b = qoi**2``) and ``EntropicDeviationMeasure`` (with
    ``weights_a = exp(alpha*qoi)``, ``weights_b = qoi``).

    Returns ``None`` when no fused path is available for this backend —
    the caller must then fall back to the legacy jacobian_matrix + einsum
    path.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    callable or None
        Implementation with signature
        (shapes, obs, latent_samples, base_variances, design_weights,
         qwl_ratio, weights_a, weights_b, bkd) -> (part_a, part_b)
        or ``None`` if no fused impl exists for this backend.
    """
    if _is_numpy(bkd) and HAS_NUMBA:

        def impl(
            shapes: Array,
            obs: Array,
            latent_samples: Optional[Array],
            base_variances: Array,
            design_weights: Array,
            qwl_ratio: Array,
            weights_a: Array,
            weights_b: Array,
            bkd: Backend[Array],
        ) -> Tuple[Array, Array]:
            has_latent = latent_samples is not None
            if not has_latent:
                latent_samples_np = np.zeros_like(obs)
            else:
                latent_samples_np = latent_samples

            # Transpose to the contiguous layout the kernel expects.
            shapes_ik = np.ascontiguousarray(shapes.T)                # (ninner, nobs)
            obs_jk = np.ascontiguousarray(obs.T)                      # (nouter, nobs)
            latent_jk = np.ascontiguousarray(latent_samples_np.T)
            weights_a_qi = np.ascontiguousarray(weights_a.T)          # (npred, ninner)
            weights_b_qi = np.ascontiguousarray(weights_b.T)

            return fused_weighted_jacobian_numba(
                shapes_ik,
                obs_jk,
                latent_jk,
                base_variances,
                design_weights,
                qwl_ratio,
                weights_a_qi,
                weights_b_qi,
                has_latent,
            )

        return impl

    return None
