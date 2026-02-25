"""Model factories for multi-fidelity sparse grids.

Provides protocols and implementations for mapping configuration indices
to callable models, with optional timing instrumentation.

Classes
-------
- ModelFactoryProtocol: Protocol for model factories
- DictModelFactory: Factory backed by a dict of callables
- TimedModelFactory: Wraps a factory with per-config timing
"""

import time
from typing import Callable, Dict, Protocol, runtime_checkable

from pyapprox.interface.functions.timing import (
    FunctionTimer,
)
from pyapprox.surrogates.sparsegrids.candidate_info import ConfigIdx


@runtime_checkable
class ModelFactoryProtocol(Protocol):
    """Protocol for model factories that map config indices to callables."""

    def get_model(self, config_idx: ConfigIdx) -> Callable:
        """Return callable model for a config.

        The returned callable should accept an Array of shape
        (nvars, nsamples) and return an Array of shape (nqoi, nsamples).

        Parameters
        ----------
        config_idx : ConfigIdx
            Configuration index (tuple of ints).

        Returns
        -------
        Callable
            Model function for the given configuration.
        """
        ...


class DictModelFactory:
    """Model factory backed by a dictionary of callables.

    Parameters
    ----------
    models : Dict[ConfigIdx, Callable]
        Mapping from config index to model callable.
    """

    def __init__(self, models: Dict[ConfigIdx, Callable]) -> None:
        self._models = models

    def get_model(self, config_idx: ConfigIdx) -> Callable:
        """Return model for config_idx.

        Raises
        ------
        KeyError
            If no model is registered for config_idx.
        """
        if config_idx not in self._models:
            raise KeyError(f"No model for config_idx={config_idx}")
        return self._models[config_idx]

    def __repr__(self) -> str:
        return f"DictModelFactory(configs={list(self._models.keys())})"


class TimedModelFactory:
    """Wraps a ModelFactoryProtocol, timing every model evaluation.

    Maintains one FunctionTimer per config_idx. Each call to get_model()
    returns a timed wrapper so evaluations are automatically timed.
    Access per-config timers via timer(config_idx) or all_timers().

    Parameters
    ----------
    base : ModelFactoryProtocol
        The underlying model factory.
    """

    def __init__(self, base: ModelFactoryProtocol) -> None:
        self._base = base
        self._timers: Dict[ConfigIdx, FunctionTimer] = {}

    def get_model(self, config_idx: ConfigIdx) -> Callable:
        """Return a timed wrapper around the base model.

        Each invocation of the returned callable records its elapsed
        time and number of evaluations in the per-config FunctionTimer.
        """
        if config_idx not in self._timers:
            self._timers[config_idx] = FunctionTimer()
        fn_timer = self._timers[config_idx]
        model = self._base.get_model(config_idx)

        def timed_model(samples):  # type: ignore[no-untyped-def]
            n_evals = samples.shape[1]
            t0 = time.perf_counter()
            result = model(samples)
            fn_timer.get("__call__").record(
                time.perf_counter() - t0, n_evals
            )
            return result

        return timed_model

    def timer(self, config_idx: ConfigIdx) -> FunctionTimer:
        """Get timer for config_idx. Creates one if it doesn't exist."""
        if config_idx not in self._timers:
            self._timers[config_idx] = FunctionTimer()
        return self._timers[config_idx]

    def all_timers(self) -> Dict[ConfigIdx, FunctionTimer]:
        """Return a copy of all per-config timers."""
        return dict(self._timers)

    def __repr__(self) -> str:
        return f"TimedModelFactory(configs={list(self._timers.keys())})"
