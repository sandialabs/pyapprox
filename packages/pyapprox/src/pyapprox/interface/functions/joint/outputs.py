"""Observations and targets evaluated at one set of samples."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class JointOutputs(Generic[Array]):
    """Observations and target blocks at the same samples.

    Parameters
    ----------
    targets : tuple[Array, ...]
        One array per target block, in the evaluator's order. Block ``k``
        has shape (target_sizes[k], nsamples).
    observations : Array
        Noise-free observations. Shape: (nobs, nsamples)

    Raises
    ------
    ValueError
        If an array is not 2D or the blocks disagree on ``nsamples``.
    """

    targets: tuple[Array, ...]
    observations: Array

    def __post_init__(self) -> None:
        if self.observations.ndim != 2:
            raise ValueError(
                f"observations must be 2D, got shape {tuple(self.observations.shape)}"
            )
        nsamples = self.observations.shape[1]
        for ii, target in enumerate(self.targets):
            if target.ndim != 2 or target.shape[1] != nsamples:
                raise ValueError(
                    f"target {ii} must have shape (*, {nsamples}), got "
                    f"{tuple(target.shape)}"
                )

    def nsamples(self) -> int:
        """Number of samples, the shared column count."""
        return int(self.observations.shape[1])
