"""Samplers for stochastic flow matching training.

Provides source and target samplers used by ``TorchSGDFitter``.
All samplers produce torch tensors for direct use in the training loop.
"""

from typing import Protocol, Tuple

import numpy as np
import torch


class SourceSamplerProtocol(Protocol):
    """Protocol for source distribution samplers."""

    def sample_x0(self, n: int) -> torch.Tensor:
        """Draw n source samples.

        Returns
        -------
        torch.Tensor
            Shape ``(d, n)``.
        """
        ...


class TargetSamplerProtocol(Protocol):
    """Protocol for target distribution samplers."""

    def sample_x1(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n target samples with weights.

        Returns
        -------
        samples : torch.Tensor
            Shape ``(d, n)``.
        weights : torch.Tensor
            Shape ``(n,)``.
        """
        ...


class GaussianSourceSampler:
    """Standard normal source sampler.

    Parameters
    ----------
    d : int
        Spatial dimension.
    dtype : torch.dtype
        Tensor dtype.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        d: int = 1,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ) -> None:
        self._d = d
        self._dtype = dtype
        self._rng = np.random.RandomState(seed)

    def sample_x0(self, n: int) -> torch.Tensor:
        """Draw n samples from N(0, I).

        Returns
        -------
        torch.Tensor
            Shape ``(d, n)``.
        """
        return torch.as_tensor(
            self._rng.randn(self._d, n), dtype=self._dtype
        )


class WeightedEmpiricalSampler:
    """Sample-with-replacement from a weighted empirical distribution.

    Parameters
    ----------
    x1_samples : torch.Tensor or array-like
        Target samples, shape ``(d, n_pool)``.
    weights : torch.Tensor or array-like
        Sample weights, shape ``(n_pool,)``. Normalized internally.
    dtype : torch.dtype
        Tensor dtype.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        x1_samples: torch.Tensor,
        weights: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ) -> None:
        self._x1 = torch.as_tensor(x1_samples, dtype=dtype)
        w = torch.as_tensor(weights, dtype=dtype)
        self._weights = w / w.sum()
        self._rng = np.random.RandomState(seed)

    def sample_x1(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw n samples with replacement, weighted by importance weights.

        Returns
        -------
        samples : torch.Tensor
            Shape ``(d, n)``.
        weights : torch.Tensor
            Uniform weights ``1/n``, shape ``(n,)``.
        """
        probs = self._weights.detach().cpu().numpy()
        idx = self._rng.choice(len(probs), size=n, replace=True, p=probs)
        samples = self._x1[:, idx]
        w = torch.full((n,), 1.0 / n, dtype=samples.dtype)
        return samples, w
