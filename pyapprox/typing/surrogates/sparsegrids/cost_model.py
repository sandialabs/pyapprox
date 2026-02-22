"""Cost models for multi-fidelity sparse grids.

Provides per-sample cost estimates for different model configurations,
used by CostWeightedIndicator to normalize error indicators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyapprox.typing.surrogates.sparsegrids.model_factory import (
        TimedModelFactory,
    )

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


class MeasuredCostModel:
    """Cost model that reads measured wall times from a TimedModelFactory.

    Returns median per-sample time for configs that have been evaluated.
    Returns 1.0 for configs with no measurements yet.

    Parameters
    ----------
    timed_factory : TimedModelFactory
        Timed model factory whose per-config timers provide cost data.
    """

    def __init__(self, timed_factory: TimedModelFactory) -> None:
        self._timed_factory = timed_factory

    def __call__(self, config_idx: ConfigIdx) -> float:
        method_timer = self._timed_factory.timer(config_idx).get("__call__")
        if method_timer.call_count() == 0:
            return 1.0
        return method_timer.median()

    def __repr__(self) -> str:
        return "MeasuredCostModel()"
