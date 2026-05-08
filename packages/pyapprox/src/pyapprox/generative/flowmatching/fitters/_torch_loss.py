"""Shared torch loss for flow matching fitters."""

import torch


def weighted_mse(
    v_pred: torch.Tensor,
    u_target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE between predicted and target velocity.

    Parameters
    ----------
    v_pred : torch.Tensor
        Predicted velocity, shape ``(nqoi, ns)``.
    u_target : torch.Tensor
        Target velocity, shape ``(nqoi, ns)``.
    weights : torch.Tensor
        Sample weights, shape ``(ns,)``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    diff = v_pred - u_target
    return (weights * torch.sum(diff * diff, dim=0)).sum()
