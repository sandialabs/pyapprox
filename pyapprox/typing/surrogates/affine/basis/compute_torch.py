"""
Torch-native implementations of MultiIndexBasis tensor product assembly.

These functions call torch.* directly instead of going through the Backend
abstraction, enabling clean torch.compile tracing without graph breaks
from bkd.* method dispatch.

Each function mirrors the corresponding function in compute.py but uses
torch.gather for index selection (cleaner compile tracing than fancy indexing)
and torch operations directly. Preserves autograd graph throughout.
"""

from typing import List

import torch


def _gather_indices(
    vals_1d: List[torch.Tensor],
    indices: torch.Tensor,
) -> List[torch.Tensor]:
    """Gather selected 1D values using torch.gather.

    Parameters
    ----------
    vals_1d : List[torch.Tensor]
        Univariate basis values. vals_1d[d] has shape (nsamples, nterms_1d_d).
    indices : torch.Tensor
        Multi-indices. Shape: (nvars, nterms). Integer dtype.

    Returns
    -------
    List[torch.Tensor]
        Each element has shape (nsamples, nterms).
    """
    nvars = len(vals_1d)
    nsamples = vals_1d[0].shape[0]
    gathered = []
    for dd in range(nvars):
        idx = indices[dd, :].unsqueeze(0).expand(nsamples, -1).long()
        gathered.append(torch.gather(vals_1d[dd], 1, idx))
    return gathered


def basis_eval_torch(
    vals_1d: List[torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate tensor product basis using torch operations.

    Parameters
    ----------
    vals_1d : List[torch.Tensor]
        Univariate basis values. vals_1d[d] has shape (nsamples, nterms_1d_d).
    indices : torch.Tensor
        Multi-indices. Shape: (nvars, nterms). Integer dtype.

    Returns
    -------
    torch.Tensor
        Basis matrix. Shape: (nsamples, nterms).
    """
    gathered = _gather_indices(vals_1d, indices)
    stacked = torch.stack(gathered, dim=0)  # (nvars, nsamples, nterms)
    return torch.prod(stacked, dim=0)  # (nsamples, nterms)


def basis_jacobian_torch(
    vals_1d: List[torch.Tensor],
    derivs_1d: List[torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Jacobian of tensor product basis using prefix/suffix products.

    Parameters
    ----------
    vals_1d : List[torch.Tensor]
        Univariate basis values. vals_1d[d] shape: (nsamples, nterms_1d_d).
    derivs_1d : List[torch.Tensor]
        Univariate first derivatives. derivs_1d[d] shape: (nsamples, nterms_1d_d).
    indices : torch.Tensor
        Multi-indices. Shape: (nvars, nterms). Integer dtype.

    Returns
    -------
    torch.Tensor
        Jacobian. Shape: (nsamples, nterms, nvars).
    """
    nvars = len(vals_1d)
    gathered_vals = _gather_indices(vals_1d, indices)
    gathered_derivs = _gather_indices(derivs_1d, indices)

    nsamples = vals_1d[0].shape[0]
    nterms = indices.shape[1]
    ones = torch.ones(
        nsamples, nterms, dtype=vals_1d[0].dtype, device=vals_1d[0].device,
    )

    prefix: List[torch.Tensor] = [torch.empty(0)] * nvars
    prefix[0] = ones
    for dd in range(1, nvars):
        prefix[dd] = prefix[dd - 1] * gathered_vals[dd - 1]

    suffix: List[torch.Tensor] = [torch.empty(0)] * nvars
    suffix[nvars - 1] = ones
    for dd in range(nvars - 2, -1, -1):
        suffix[dd] = suffix[dd + 1] * gathered_vals[dd + 1]

    jac_list = []
    for dd in range(nvars):
        jac_list.append(gathered_derivs[dd] * prefix[dd] * suffix[dd])

    return torch.stack(jac_list, dim=-1)  # (nsamples, nterms, nvars)


def basis_hessian_torch(
    vals_1d: List[torch.Tensor],
    derivs_1d: List[torch.Tensor],
    hess_1d: List[torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Hessian of tensor product basis.

    Parameters
    ----------
    vals_1d : List[torch.Tensor]
        Univariate basis values. vals_1d[d] shape: (nsamples, nterms_1d_d).
    derivs_1d : List[torch.Tensor]
        Univariate first derivatives. derivs_1d[d] shape: (nsamples, nterms_1d_d).
    hess_1d : List[torch.Tensor]
        Univariate second derivatives. hess_1d[d] shape: (nsamples, nterms_1d_d).
    indices : torch.Tensor
        Multi-indices. Shape: (nvars, nterms). Integer dtype.

    Returns
    -------
    torch.Tensor
        Hessian. Shape: (nsamples, nterms, nvars, nvars).
    """
    nvars = len(vals_1d)
    gathered_vals = _gather_indices(vals_1d, indices)
    gathered_derivs = _gather_indices(derivs_1d, indices)
    gathered_hess = _gather_indices(hess_1d, indices)

    nsamples = vals_1d[0].shape[0]
    nterms = indices.shape[1]
    ones = torch.ones(
        nsamples, nterms, dtype=vals_1d[0].dtype, device=vals_1d[0].device,
    )

    # Prefix/suffix for leave-one-out products
    prefix: List[torch.Tensor] = [torch.empty(0)] * nvars
    prefix[0] = ones
    for dd in range(1, nvars):
        prefix[dd] = prefix[dd - 1] * gathered_vals[dd - 1]

    suffix: List[torch.Tensor] = [torch.empty(0)] * nvars
    suffix[nvars - 1] = ones
    for dd in range(nvars - 2, -1, -1):
        suffix[dd] = suffix[dd + 1] * gathered_vals[dd + 1]

    result = torch.zeros(
        nsamples, nterms, nvars, nvars,
        dtype=vals_1d[0].dtype, device=vals_1d[0].device,
    )

    for dd in range(nvars):
        leave_one_out_dd = prefix[dd] * suffix[dd]

        # Diagonal: hess_1d[dd] * leave_one_out[dd]
        result[:, :, dd, dd] = gathered_hess[dd] * leave_one_out_dd

        # Off-diagonal: derivs[dd] * derivs[kk] * leave_two_out[dd, kk]
        for kk in range(dd + 1, nvars):
            hess_dk = gathered_derivs[dd] * gathered_derivs[kk]
            for ll in range(nvars):
                if ll != dd and ll != kk:
                    hess_dk = hess_dk * gathered_vals[ll]
            result[:, :, dd, kk] = hess_dk
            result[:, :, kk, dd] = hess_dk

    return result
