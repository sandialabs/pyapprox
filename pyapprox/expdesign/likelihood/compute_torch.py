"""
Torch-native implementations of OED likelihood computations.

These functions call torch.* directly instead of going through the Backend
abstraction, enabling clean torch.compile tracing without graph breaks
from bkd.* method dispatch.

Each function mirrors the corresponding function in compute.py but uses
torch operations directly. The has_latent: bool flag replaces Optional[Array]
to avoid suboptimal torch.compile handling of Optional types.
"""

import math

import torch


def logpdf_matrix_torch(
    shapes: torch.Tensor,
    obs: torch.Tensor,
    base_variances: torch.Tensor,
    design_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute log-likelihood matrix using torch operations directly.

    Parameters
    ----------
    shapes : torch.Tensor
        Model outputs for inner samples. Shape: (nobs, ninner)
    obs : torch.Tensor
        Observations for outer samples. Shape: (nobs, nouter)
    base_variances : torch.Tensor
        Base noise variances. Shape: (nobs,)
    design_weights : torch.Tensor
        Design weights. Shape: (nobs, 1)

    Returns
    -------
    torch.Tensor
        Log-likelihood matrix. Shape: (ninner, nouter)
    """
    nobs = shapes.shape[0]
    inv_var = design_weights[:, 0] / base_variances

    # Log normalization as scalar tensor (preserves autograd graph)
    log_det = torch.sum(torch.log(base_variances)) - torch.sum(
        torch.log(design_weights)
    )
    log_norm = (
        torch.tensor(
            -0.5 * nobs * math.log(2 * math.pi),
            dtype=design_weights.dtype,
            device=design_weights.device,
        )
        - 0.5 * log_det
    )

    # residuals: (nobs, ninner, nouter)
    residuals = obs[:, None, :] - shapes[:, :, None]
    sqrt_inv_var = torch.sqrt(inv_var)
    scaled_res = sqrt_inv_var[:, None, None] * residuals
    squared_dist = torch.einsum("ijk,ijk->jk", scaled_res, scaled_res)

    return log_norm - 0.5 * squared_dist


def jacobian_matrix_torch(
    shapes: torch.Tensor,
    obs: torch.Tensor,
    latent_samples: torch.Tensor,
    base_variances: torch.Tensor,
    design_weights: torch.Tensor,
    has_latent: bool,
) -> torch.Tensor:
    """Compute Jacobian of log-likelihood matrix w.r.t. design weights.

    Parameters
    ----------
    shapes : torch.Tensor
        Model outputs for inner samples. Shape: (nobs, ninner)
    obs : torch.Tensor
        Observations for outer samples. Shape: (nobs, nouter)
    latent_samples : torch.Tensor
        Latent samples for reparameterization. Shape: (nobs, nouter)
        Ignored if has_latent is False.
    base_variances : torch.Tensor
        Base noise variances. Shape: (nobs,)
    design_weights : torch.Tensor
        Design weights. Shape: (nobs, 1)
    has_latent : bool
        Whether latent_samples should be used. Python-level control flow
        creates two static branches for torch.compile.

    Returns
    -------
    torch.Tensor
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
    if has_latent:
        reparam_term = 0.5 * (
            residuals
            * latent_samples[:, None, :]
            / (
                torch.sqrt(base_variances[:, None, None])
                * torch.sqrt(design_weights[:, :, None])
            )
        )
        jac = jac + reparam_term

    # Transpose from (nobs, ninner, nouter) to (ninner, nouter, nobs)
    return jac.permute(1, 2, 0)


def evidence_jacobian_torch(
    loglike_jacobian: torch.Tensor,
    quad_weighted_like: torch.Tensor,
) -> torch.Tensor:
    """Compute evidence jacobian via einsum contraction.

    Parameters
    ----------
    loglike_jacobian : torch.Tensor
        Jacobian of log-likelihood matrix. Shape: (ninner, nouter, nobs)
    quad_weighted_like : torch.Tensor
        Quadrature-weighted likelihoods. Shape: (ninner, nouter)

    Returns
    -------
    torch.Tensor
        Evidence jacobian. Shape: (nouter, nobs)
    """
    return torch.einsum("io,iok->ok", quad_weighted_like, loglike_jacobian)
