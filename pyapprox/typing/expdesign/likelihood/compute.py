"""
Vectorized free-function implementations of OED likelihood computations.

These functions use only Backend[Array] operations, making them compatible
with both NumPy and PyTorch backends. They are also torch.compile-friendly
for future PyTorch acceleration.

Each function corresponds to a method on GaussianOEDInnerLoopLikelihood
but takes all data as explicit arguments rather than reading from self.
"""
import math
from typing import Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


def compute_log_normalization(
    base_variances: Array,
    design_weights: Array,
    nobs: int,
    bkd: Backend[Array],
) -> float:
    """Compute log normalization constant for Gaussian likelihood."""
    log_det = float(
        bkd.sum(bkd.log(base_variances))
        - bkd.sum(bkd.log(design_weights))
    )
    return float(-0.5 * nobs * math.log(2 * math.pi) - 0.5 * log_det)


def logpdf_matrix_vectorized(
    shapes: Array,
    obs: Array,
    base_variances: Array,
    design_weights: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute log-likelihood matrix using vectorized backend operations.

    Parameters
    ----------
    shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    obs : Array
        Observations for outer samples. Shape: (nobs, nouter)
    base_variances : Array
        Base noise variances. Shape: (nobs,)
    design_weights : Array
        Design weights. Shape: (nobs, 1)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Log-likelihood matrix. Shape: (ninner, nouter)
    """
    nobs = shapes.shape[0]
    inv_var = design_weights[:, 0] / base_variances
    log_norm = compute_log_normalization(
        base_variances, design_weights, nobs, bkd
    )

    # residuals: (nobs, ninner, nouter)
    residuals = obs[:, None, :] - shapes[:, :, None]
    sqrt_inv_var = bkd.sqrt(inv_var)
    scaled_res = sqrt_inv_var[:, None, None] * residuals
    squared_dist = bkd.einsum("ijk,ijk->jk", scaled_res, scaled_res)

    return log_norm - 0.5 * squared_dist


def jacobian_matrix_vectorized(
    shapes: Array,
    obs: Array,
    latent_samples: Optional[Array],
    base_variances: Array,
    design_weights: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute Jacobian of log-likelihood matrix w.r.t. design weights.

    Parameters
    ----------
    shapes : Array
        Model outputs for inner samples. Shape: (nobs, ninner)
    obs : Array
        Observations for outer samples. Shape: (nobs, nouter)
    latent_samples : Array or None
        Latent samples for reparameterization. Shape: (nobs, nouter)
    base_variances : Array
        Base noise variances. Shape: (nobs,)
    design_weights : Array
        Design weights. Shape: (nobs, 1)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Jacobian tensor. Shape: (ninner, nouter, nobs)
    """
    # residuals: (nobs, ninner, nouter)
    residuals = obs[:, None, :] - shapes[:, :, None]

    # Component 1: Quadratic term
    jac = -0.5 * residuals**2 / base_variances[:, None, None]

    # Component 2: Determinant term
    det_term = 0.5 / design_weights[:, :, None]  # (nobs, 1, 1)
    jac = jac + det_term

    # Component 3: Reparameterization term
    if latent_samples is not None:
        reparam_term = 0.5 * (
            residuals
            * latent_samples[:, None, :]
            / (
                bkd.sqrt(base_variances[:, None, None])
                * bkd.sqrt(design_weights[:, :, None])
            )
        )
        jac = jac + reparam_term

    # Transpose from (nobs, ninner, nouter) to (ninner, nouter, nobs)
    return bkd.transpose(jac, (1, 2, 0))


def evidence_jacobian_vectorized(
    loglike_jacobian: Array,
    quad_weighted_like: Array,
    bkd: Backend[Array],
) -> Array:
    """Compute evidence jacobian via einsum contraction.

    Parameters
    ----------
    loglike_jacobian : Array
        Jacobian of log-likelihood matrix. Shape: (ninner, nouter, nobs)
    quad_weighted_like : Array
        Quadrature-weighted likelihoods. Shape: (ninner, nouter)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Evidence jacobian. Shape: (nouter, nobs)
    """
    return bkd.einsum("io,iok->ok", quad_weighted_like, loglike_jacobian)
