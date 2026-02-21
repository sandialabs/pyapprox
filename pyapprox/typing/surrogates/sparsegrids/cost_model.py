"""Cost models for multi-fidelity sparse grids.

Provides per-sample cost estimates for different model configurations,
used by CostWeightedIndicator to normalize error indicators.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.surrogates.sparsegrids.candidate_info import ConfigIdx


@runtime_checkable
class CostModelProtocol(Protocol):
    """Protocol for model cost estimation."""

    def __call__(self, config_idx: ConfigIdx) -> float:
        """Return per-sample evaluation cost for a config.

        Parameters
        ----------
        config_idx : ConfigIdx
            Configuration index (tuple of ints).

        Returns
        -------
        float
            Cost per sample (must be positive).
        """
        ...


class ConstantCostModel:
    """All configurations have unit cost.

    Useful for single-fidelity grids or when costs are unknown.
    """

    def __call__(self, config_idx: ConfigIdx) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "ConstantCostModel()"


class ExponentialConfigCostModel:
    """Cost = base^(sum of config index components).

    Parameters
    ----------
    base : float
        Exponential base. Default: 10.0.
    """

    def __init__(self, base: float = 10.0) -> None:
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        self._base = base

    def __call__(self, config_idx: ConfigIdx) -> float:
        return self._base ** sum(config_idx)

    def base(self) -> float:
        """Return the exponential base."""
        return self._base

    def __repr__(self) -> str:
        return f"ExponentialConfigCostModel(base={self._base})"
