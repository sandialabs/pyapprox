"""Torch-native barycentric Lagrange polynomial evaluation and derivatives.

Uses pure torch operations (no bkd.* methods) so that torch.compile
can trace and optimize the computation graph without graph breaks.
"""

from typing import Any

import torch


def lagrange_eval_torch(
    abscissa: torch.Tensor,
    samples: torch.Tensor,
    bary_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Lagrange basis polynomials via barycentric formula.

    Parameters
    ----------
    abscissa : torch.Tensor
        Interpolation nodes, shape (nabscissa,).
    samples : torch.Tensor
        Evaluation points, shape (nsamples,).
    bary_weights : torch.Tensor
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    torch.Tensor
        Basis values, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    if nabscissa == 1:
        return torch.ones(
            (samples.shape[0], 1),
            dtype=samples.dtype,
            device=samples.device,
        )

    # Differences: (nsamples, nabscissa)
    numers = samples[:, None] - abscissa[None, :]

    # Detect exact node hits
    is_node = numers == 0.0

    # Replace zeros with 1.0 for safe product computation
    safe_numers = torch.where(is_node, 1.0, numers)

    # P(x) with safe numerators: (nsamples, 1)
    full_product = torch.prod(safe_numers, dim=1, keepdim=True)

    # L_j(x) = w_j * P(x) / (x - x_j)
    values = bary_weights[None, :] * full_product / safe_numers

    # Override rows where sample coincides with a node
    any_hit = torch.any(is_node, dim=1)  # (nsamples,)
    any_hit_2d = any_hit[:, None]  # (nsamples, 1)
    node_values = torch.where(is_node, 1.0, 0.0)
    values = torch.where(any_hit_2d, node_values, values)

    return values


def _node_deriv_matrices_torch(
    abscissa: torch.Tensor,
    bary_weights: torch.Tensor,
) -> tuple[Any, ...]:
    """Precompute first and second derivative matrices at nodes.

    Returns (D1, D2) where D1[m,j] = L'_j(x_m), D2[m,j] = L''_j(x_m).
    """
    n = abscissa.shape[0]
    # node_diffs[m, j] = x_m - x_j
    node_diffs = abscissa[:, None] - abscissa[None, :]
    is_diag = torch.eye(n, dtype=torch.bool, device=abscissa.device)
    safe_diffs = torch.where(is_diag, 1.0, node_diffs)

    # D1[m, j] = w_j / (w_m * (x_m - x_j)) for j != m
    D1_off = bary_weights[None, :] / (bary_weights[:, None] * safe_diffs)
    D1_off = torch.where(is_diag, 0.0, D1_off)
    D1_diag = -torch.sum(D1_off, dim=1)  # (n,)
    D1 = D1_off + torch.diag(D1_diag)

    # D2[m, j] = 2 * D1[m,j] * (D1[m,m] - 1/(x_m - x_j)) for j != m
    inv_diffs = torch.where(is_diag, 0.0, 1.0 / safe_diffs)
    D2_off = 2.0 * D1_off * (D1_diag[:, None] - inv_diffs)
    D2_diag = -torch.sum(D2_off, dim=1)
    D2 = D2_off + torch.diag(D2_diag)

    return D1, D2


def lagrange_jacobian_torch(
    abscissa: torch.Tensor,
    samples: torch.Tensor,
    bary_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate first derivatives of Lagrange basis polynomials.

    Uses L'_j(x) = L_j(x) * S_j(x) where S_j = sum_{k!=j} 1/(x-x_k).

    Parameters
    ----------
    abscissa : torch.Tensor
        Interpolation nodes, shape (nabscissa,).
    samples : torch.Tensor
        Evaluation points, shape (nsamples,).
    bary_weights : torch.Tensor
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    torch.Tensor
        First derivatives, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    nsamples = samples.shape[0]
    if nabscissa == 1:
        return torch.zeros((nsamples, 1), dtype=samples.dtype, device=samples.device)

    diffs = samples[:, None] - abscissa[None, :]

    # Tolerance-based near-node detection
    tol = 1e-12 * (1.0 + torch.max(torch.abs(abscissa)))
    is_near = torch.abs(diffs) < tol
    safe_diffs = torch.where(is_near, 1.0, diffs)

    basis_vals = lagrange_eval_torch(abscissa, samples, bary_weights)

    inv_diffs = torch.where(is_near, 0.0, 1.0 / safe_diffs)
    total_sum = torch.sum(inv_diffs, dim=1, keepdim=True)
    S = total_sum - inv_diffs

    derivs = basis_vals * S

    # Near-node case
    any_hit = torch.any(is_near, dim=1)
    any_hit_2d = any_hit[:, None]
    D1, _ = _node_deriv_matrices_torch(abscissa, bary_weights)
    is_near_float = torch.where(is_near, 1.0, 0.0)
    node_result = torch.matmul(is_near_float, D1)
    derivs = torch.where(any_hit_2d, node_result, derivs)

    return derivs


def lagrange_hessian_torch(
    abscissa: torch.Tensor,
    samples: torch.Tensor,
    bary_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate second derivatives of Lagrange basis polynomials.

    Uses L''_j(x) = L_j(x) * (S_j^2 - T_j) where
    S_j = sum_{k!=j} 1/(x-x_k), T_j = sum_{k!=j} 1/(x-x_k)^2.

    Parameters
    ----------
    abscissa : torch.Tensor
        Interpolation nodes, shape (nabscissa,).
    samples : torch.Tensor
        Evaluation points, shape (nsamples,).
    bary_weights : torch.Tensor
        Precomputed barycentric weights, shape (nabscissa,).

    Returns
    -------
    torch.Tensor
        Second derivatives, shape (nsamples, nabscissa).
    """
    nabscissa = abscissa.shape[0]
    nsamples = samples.shape[0]
    if nabscissa == 1:
        return torch.zeros((nsamples, 1), dtype=samples.dtype, device=samples.device)

    diffs = samples[:, None] - abscissa[None, :]

    tol = 1e-12 * (1.0 + torch.max(torch.abs(abscissa)))
    is_near = torch.abs(diffs) < tol
    safe_diffs = torch.where(is_near, 1.0, diffs)

    basis_vals = lagrange_eval_torch(abscissa, samples, bary_weights)

    inv_diffs = torch.where(is_near, 0.0, 1.0 / safe_diffs)
    total_sum = torch.sum(inv_diffs, dim=1, keepdim=True)
    S = total_sum - inv_diffs

    inv_diffs_sq = inv_diffs * inv_diffs
    total_sum_sq = torch.sum(inv_diffs_sq, dim=1, keepdim=True)
    T = total_sum_sq - inv_diffs_sq

    derivs = basis_vals * (S * S - T)

    # Near-node case
    any_hit = torch.any(is_near, dim=1)
    any_hit_2d = any_hit[:, None]
    _, D2 = _node_deriv_matrices_torch(abscissa, bary_weights)
    is_near_float = torch.where(is_near, 1.0, 0.0)
    node_result = torch.matmul(is_near_float, D2)
    derivs = torch.where(any_hit_2d, node_result, derivs)

    return derivs
