"""Torch-native barycentric Lagrange polynomial evaluation.

Uses pure torch operations (no bkd.* methods) so that torch.compile
can trace and optimize the computation graph without graph breaks.
"""

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
